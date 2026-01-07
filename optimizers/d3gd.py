import torch
import torch.nn.functional as F
from .base import DecentralizedOptimizer

class D3GD(DecentralizedOptimizer):
    def __init__(self, config, nodes, topology_manager):
        super().__init__(config, nodes, topology_manager)
        
        params = config.algorithm.d3gd
        self.delta = params.delta
        self.topo_lr = params.topo_lr
        self.lr = config.training.global_lr
        
        # Initial Topology A0 (Static Anchor)
        self.A0 = self.topo.get_weights().clone()
        
        # A_bar (The learnable parameter) - initialized to A0
        # We need gradients for this, but we implement manual projected gradient
        self.A_bar = self.A0.clone()
        
        # Y vector for Di-DGD
        self.y = torch.ones(self.num_nodes, 1, device=self.device)
        
        # Trackers z and q
        # z tracks weighted params, q tracks weighted gradients
        # Initialize z0 = theta0, q0 = grad0
        self.theta_curr = self._collect_node_params() # (N, d)
        self.grads_curr = self._collect_gradients()   # (N, d)
        
        self.z = self.theta_curr.clone()
        self.q = self.grads_curr.clone()

    def _project_row_stochastic(self, matrix):
        """
        Projects rows onto the simplex, respecting physical constraints.
        """
        # 1. Mask with physical graph (Ensure 0 stays 0)
        mask = self.topo.physical_mask + torch.eye(self.num_nodes, device=self.device)
        matrix = matrix * mask
        
        # 2. Simplex Projection (Sorting method)
        # Simple alternative: Relu and Normalize (faster, approx)
        matrix = torch.relu(matrix)
        row_sums = matrix.sum(dim=1, keepdim=True) + 1e-8
        return matrix / row_sums

    def step(self):
        # --- 1. Compute Composite Adjacency A^k ---
        # A^k = (1 - delta) * A_bar + delta * A0
        A_k = (1 - self.delta) * self.A_bar + self.delta * self.A0
        
        # --- 2. Di-DGD Step (Standard) ---
        theta_mixed = self.neighbor_mix(self.theta_curr, A_k)
        y_mixed = self.neighbor_mix(self.y, A_k.T)
        
        y_denom = y_mixed * self.num_nodes
        grad_correction = self.grads_curr / (y_denom + 1e-8)
        
        theta_next = theta_mixed - self.lr * grad_correction
        
        # --- 3. Topology Update (The Core D3GD Logic) ---
        # We need to compute gradients for A_bar row by row
        # This corresponds to Eq 9 / Eq 14 in the paper
        
        # Pre-compute terms to vectorize
        # Term 1: (gamma * (1-delta) / (n * y_i)) * q_i - z_i
        # Shape (N, d)
        term1_coeff = (self.lr * (1 - self.delta)) / (y_denom + 1e-8)
        term1 = term1_coeff * self.q - self.z
        
        # Term 2: Theta_mixed (which is A * Theta) - correction term
        # The paper equation is complex. Let's simplify based on the "Design Function J"
        # J ~ || (A - 1 pi^T) Theta ||^2
        # Gradient of J w.r.t A: 2 * (A*Theta - Mean) * Theta^T
        # We use the explicit decentralized gradient approximation from snippet Eq (9)
        
        # G_i (Gradient for row i) approx:
        # 2 * Sum_j ( Theta_j^T * Term1_i ) ... this looks like matrix mult
        
        # Vectorized Gradient Computation:
        # Grad_A = 2 * [ Term1 @ Theta^T + Term2 @ Theta^T ]
        # Where Term2 = (A*Theta - term1_coeff * Grad_i)
        
        term2 = theta_mixed - (term1_coeff * self.grads_curr)
        
        # Combine: Grad_A = 2 * (Term1 + Term2) @ Theta^T
        # Note: In the paper, it's specific to neighbors.
        # (N, d) @ (d, N) -> (N, N)
        grad_A = 2 * torch.matmul(term1 + term2, self.theta_curr.T)
        
        # Update A_bar
        # A_bar_new = Proj(A_bar - eta * Grad_A)
        self.A_bar = self._project_row_stochastic(
            self.A_bar - self.topo_lr * grad_A
        )
        
        # --- 4. Tracker Updates ---
        # We need Theta_next and Grad_next
        self._distribute_node_params(theta_next) # Apply updates to nodes
        
        # Compute new gradients at Theta_next
        grads_next = self._collect_gradients()
        
        # Update z and q (Eq 15, 16)
        # z_next = A * z + (theta_next - theta_curr)
        z_mix = self.neighbor_mix(self.z, A_k)
        z_next = z_mix + (theta_next - self.theta_curr)
        
        # q_next = A * q + (grad_next - grad_curr)
        q_mix = self.neighbor_mix(self.q, A_k)
        q_next = q_mix + (grads_next - self.grads_curr)
        
        # --- 5. State Transition ---
        self.theta_curr = theta_next
        self.grads_curr = grads_next
        self.y = y_mixed
        self.z = z_next
        self.q = q_next
        
        self.global_step += 1
        
        # --- 6. Metrics ---
        # Recalculate Pi for the NEW A_k to measure consensus correctly
        # In dynamic graphs, the "consensus target" shifts.
        pi_k = self.topo.get_perron_vector(A_k)
        self.compute_consensus_error(perron_vector=pi_k)
        
        # Communication Volume
        # Di-DGD step + Tracker Steps (z, q) + Gradient Steps
        # A is dense-ish locally. 
        active_edges = torch.count_nonzero(A_k).item()
        # Messages: Theta (d), Y (1), Z (d), Q (d)
        vol = active_edges * (3 * theta_next.shape[1] + 1)
        self.log_metrics(0.0, vol)