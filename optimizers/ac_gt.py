import torch
import numpy as np
from .base import DecentralizedOptimizer

class AC_GT(DecentralizedOptimizer):
    def __init__(self, config, nodes, topology_manager):
        super().__init__(config, nodes, topology_manager)
        
        # Hyperparameters
        params = config.algorithm.ac_gt
        self.cycle_length = params.cycle_length
        self.pruning_fraction = params.pruning_threshold  # kappa
        self.beta = params.softmax_beta
        self.lr = config.training.global_lr
        
        # State variables
        self.y = torch.zeros(self.num_nodes, self.nodes[0].get_params().shape[0], device=self.device)
        self.grads_curr = self._collect_gradients()
        self.y = self.grads_curr.clone() # Initialize y = grad
        
        # Current Topologies (Initialized to Physical Graph Base)
        # We start with the full Metropolis-Hastings weights (no pruning at step 0)
        self.W_x = self.topo.get_weights().clone()
        self.W_y = self.topo.get_weights().clone()
        
        self.theta_curr = self._collect_node_params()

    def _run_pruning_protocol(self, values):
        """
        Implements Algorithm 1 from AC-GT paper.
        Returns a binary mask of kept edges.
        """
        N = self.num_nodes
        adj = self.topo.physical_layer.get_adjacency()
        kept_mask = torch.zeros_like(self.topo.physical_mask)
        
        # Calculate pair-wise distances (L2 norm)
        # (N, d) -> (N, N) distance matrix
        # dist[i,j] = ||v_i - v_j||
        dists = torch.cdist(values, values, p=2)
        
        # For each node, select edges to PRUNE
        # The paper says: Probability of pruning ~ exp(-beta * distance)
        # This means we are likely to prune edges with SMALL distance (similar values)
        # and keep edges with LARGE distance (influential edges).
        
        for i in range(N):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) == 0:
                continue
                
            # Calculate Pruning Probabilities
            # P(prune_j) = exp(-beta * dist_ij) / Sum(...)
            # We want to prune `fraction` of neighbors
            num_to_prune = int(len(neighbors) * self.pruning_fraction)
            num_to_keep = len(neighbors) - num_to_prune
            
            if num_to_prune == 0:
                kept_mask[i, neighbors] = 1.0
                continue

            local_dists = dists[i, neighbors]
            
            # Safe Softmax for probabilities
            # We use negative beta * dist because we want high prob for LOW dist
            logits = -self.beta * local_dists
            probs = torch.softmax(logits, dim=0)
            
            # Sample edges to PRUNE
            # Using torch.multinomial without replacement
            # Note: We need to select which ones to *KEEP* eventually
            # It's easier to select 'num_to_prune' indices to remove
            prune_indices = torch.multinomial(probs, num_to_prune, replacement=False)
            
            # Determine kept neighbors
            kept_indices = []
            for idx in range(len(neighbors)):
                if idx not in prune_indices:
                    kept_indices.append(neighbors[idx])
            
            kept_mask[i, kept_indices] = 1.0
            
        # Symmetrization (Paper Section 2.2: "Intersection" logic)
        # "Node i removes (i,j)... requires (j,i) to be removed to maintain undirectedness"
        # Mask = Mask AND Mask.T
        symmetric_mask = kept_mask * kept_mask.T
        
        # Ensure self-loops are kept
        symmetric_mask.fill_diagonal_(1.0)
        
        return symmetric_mask

    def _generate_metropolis_weights(self, mask):
        """
        Re-calculates Metropolis-Hastings weights for the pruned graph.
        """
        # Degrees of the PRUNED graph
        degrees = mask.sum(dim=1)
        
        W = torch.zeros_like(mask)
        N = self.num_nodes
        
        # Iterate only over non-zero elements for efficiency
        # (But vectorized is fine for N=100)
        
        # Broadcast degrees: deg_matrix[i, j] = max(d_i, d_j)
        deg_i = degrees.view(-1, 1).expand(N, N)
        deg_j = degrees.view(1, -1).expand(N, N)
        max_deg = torch.max(deg_i, deg_j)
        
        # W_ij = 1 / (1 + max(d_i, d_j)) for neighbors
        W = mask / (1.0 + max_deg)
        
        # Fix diagonal: W_ii = 1 - sum(W_ij for j!=i)
        # Current diagonal of W contains 1/(1+d_i), which is wrong.
        # Zero out diagonal first
        W.fill_diagonal_(0.0)
        row_sums = W.sum(dim=1)
        
        # Set new diagonal
        diag_val = 1.0 - row_sums
        W.as_strided([N], [N+1]).copy_(diag_val)
        
        return W

    def step(self):
        # --- 1. Topology Adaptation (Start of Cycle) ---
        if self.global_step % self.cycle_length == 0:
            # Prune based on X (Parameters)
            mask_x = self._run_pruning_protocol(self.theta_curr)
            self.W_x = self._generate_metropolis_weights(mask_x)
            
            # Prune based on Y (Gradient Trackers)
            mask_y = self._run_pruning_protocol(self.y)
            self.W_y = self._generate_metropolis_weights(mask_y)

        # --- 2. X Update (Gradient Tracking Form) ---
        # x_next = W_x * (x_now - alpha * y_now)
        # Note: AC-GT typically puts the mixing matrix OUTSIDE the subtraction
        # Eq 21: x_{k+1} = Q_k (x_k - alpha * y_k)

        # --- [NEW] Gradient Clipping ---
        # Clip 'y' temporarily for the update to ensure stability,
        # but DO NOT overwrite self.y as it breaks the tracking conservation property.
        y_norm = torch.norm(self.y, p=2, dim=1, keepdim=True)
        max_norm = 5.0
        clip_coef = torch.clamp(max_norm / (y_norm + 1e-6), max=1.0)
        y_clipped = self.y * clip_coef
        # -------------------------------
        
        update_direction = self.theta_curr - self.lr * self.y
        theta_next = self.neighbor_mix(update_direction, self.W_x)
        
        self._distribute_node_params(theta_next)
        
        # --- 3. Gradient Computation ---
        grads_next = self._collect_gradients()
        
        # --- 4. Y Update (Gradient Tracking Form) ---
        # y_next = W_y * y_now + (grad_next - grad_now)
        # Eq 22
        
        y_mixed = self.neighbor_mix(self.y, self.W_y)
        y_next = y_mixed + (grads_next - self.grads_curr)
        
        # --- 5. State Transition ---
        self.theta_curr = theta_next
        self.grads_curr = grads_next
        self.y = y_next
        self.global_step += 1
        
        # --- 6. Logging ---
        # Calculate Pi for W_x (the one governing parameters) for accurate error metric
        pi = self.topo.get_perron_vector(self.W_x)
        self.compute_consensus_error(perron_vector=pi)
        
        # Communication Volume
        # Edges in W_x carrying (d) params
        # Edges in W_y carrying (d) trackers
        edges_x = (torch.count_nonzero(self.W_x).item() - self.num_nodes) # Exclude self-loop
        edges_y = (torch.count_nonzero(self.W_y).item() - self.num_nodes)
        
        vol = (edges_x * self.theta_curr.shape[1]) + (edges_y * self.y.shape[1])
        self.log_metrics(0.0, vol)