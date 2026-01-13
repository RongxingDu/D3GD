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
        self.A_bar = self.A0.clone()
        
        # [FIX 1] Y vector: Initialize to sum to 1 (1/N per node)
        # This ensures n * y_i approx 1, matching the paper's scaling
        self.y = torch.ones(self.num_nodes, 1, device=self.device) / self.num_nodes
        
        # Trackers z and q
        # z tracks weighted parameters (Approximates \pi^T \theta)
        # q tracks weighted gradients
        self.theta_curr = self._collect_node_params() # (N, d)
        self.grads_curr = self._collect_gradients()   # (N, d)
        
        # [FIX 2] z must track theta values to balance the gradient term (A*theta - z)
        self.z = self.theta_curr.clone() 
        self.q = self.grads_curr.clone()

    def _project_row_stochastic(self, matrix):
        """
        Projects rows onto the simplex, respecting physical constraints.
        Executed conceptually by each agent on its own row.
        """
        # 1. Mask with physical graph (Ensure non-neighbors stay 0)
        mask = self.topo.physical_mask + torch.eye(self.num_nodes, device=self.device)
        matrix = matrix * mask
        
        # 2. Simplex Projection
        # (ReLU + Normalize is a standard efficient proxy for Simplex projection in DL)
        matrix = torch.relu(matrix)
        row_sums = matrix.sum(dim=1, keepdim=True) + 1e-12
        return matrix / row_sums

    def check_tensor(self, name, tensor, threshold=100.0, lower_bound=None):
        """Helper to catch explosions early."""
        # 1. Check for NaNs/Infs
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"\n[CRITICAL] {name} has NaNs/Infs at step {self.global_step}!")
            return True

        # 2. Check for Values exploding (too large)
        max_val = tensor.abs().max().item()
        if max_val > threshold:
            print(f"\n[WARNING] {name} is exploding! Max value: {max_val:.4f} (Threshold: {threshold})")
            return True
            
        # 3. Check for Values vanishing (too small) - mostly for 'y'
        if lower_bound is not None:
            min_val = tensor.abs().min().item()
            if min_val < lower_bound:
                print(f"\n[WARNING] {name} is vanishing! Min value: {min_val:.8f} (Lower Bound: {lower_bound})")
                return True
        return False

    def step(self):
        # 1. Compute Composite Adjacency A^k
        # Ensure delta is large enough in config (e.g., 0.1) to keep graph connected
        A_k = (1 - self.delta) * self.A_bar + self.delta * self.A0
        
        # 2. Di-DGD Step
        theta_mixed = self.neighbor_mix(self.theta_curr, A_k)
        
        # PURE IMPLEMENTATION: No clamping on y directly
        y_denom = self.y * self.num_nodes
        
        # Calculate the raw correction
        # Add epsilon ONLY to prevent strict NaN (division by zero), not to alter logic
        grad_correction = self.grads_curr / (y_denom + 1e-10)
        
        # [VALID FIX] Gradient Clipping
        # Instead of modifying y, we enforce that the total update step isn't too large.
        # This aligns with Assumption 2.3 (Bounded Gradients).
        # Calculate norm of the proposed update
        update_norm = torch.norm(grad_correction, p=2, dim=1, keepdim=True)
        
        # Clip updates that exceed a reasonable threshold (e.g., 10.0)
        # This handles the 1/y explosion without breaking the direction of the update
        max_norm = 5.0
        clip_coef = torch.clamp(max_norm / (update_norm + 1e-6), max=1.0)
        grad_correction = grad_correction * clip_coef

        theta_next = theta_mixed - self.lr * grad_correction
        
        # Update y (Standard Mixing)
        y_mixed = self.neighbor_mix(self.y, A_k.T)
        
        # 3. Topology Update
        # We also clip the components entering the topology gradient to prevent A_bar explosion
        c_i = (self.lr * (1 - self.delta)) / (y_denom + 1e-10)
        
        # Clip c_i to prevent numerical overflow in the topology gradient calculation
        c_i = torch.clamp(c_i, max=100.0) 
        
        part1 = c_i * self.q - self.z
        part2 = theta_mixed - (c_i * self.grads_curr)
        V = part1 + part2
        
        grad_A_global = 2 * torch.matmul(V, self.theta_curr.T)
        
        # Clip Topology Gradient (Standard PGD practice)
        grad_A_global = torch.clamp(grad_A_global, min=-100, max=100)
        
        # [Corrected Mask with Self-Loop]
        mask = self.topo.physical_mask + torch.eye(self.num_nodes, device=self.device)
        grad_A_local = grad_A_global * (mask > 0).float()
        
        self.A_bar = self._project_row_stochastic(
            self.A_bar - self.topo_lr * grad_A_local
        )
        
        # 4. Tracker Updates ... (Same as before)
        self._distribute_node_params(theta_next) 
        grads_next = self._collect_gradients()
        
        z_mix = self.neighbor_mix(self.z, A_k)
        z_next = z_mix + (theta_next - self.theta_curr)
        
        q_mix = self.neighbor_mix(self.q, A_k)
        q_next = q_mix + (grads_next - self.grads_curr)
        
        # 5. State Transition
        self.theta_curr = theta_next
        self.grads_curr = grads_next
        self.y = y_mixed    
        self.z = z_next     
        self.q = q_next     
        
        self.global_step += 1
        
        # Metrics... (Same as before)
        pi_k = self.topo.get_perron_vector(A_k)
        self.compute_consensus_error(perron_vector=pi_k)
        
        active_edges = torch.count_nonzero(A_k).item()
        vol = active_edges * (3 * theta_next.shape[1] + 1)
        self.log_metrics(0.0, vol)
    
    # In optimizers/d3gd.py inside class D3GD

    def get_effective_topology(self):
        """
        Returns the effective communication matrix A^k used for consensus.
        A^k = (1 - delta) * A_bar + delta * A0
        """
        A_k = (1 - self.delta) * self.A_bar + self.delta * self.A0
        return A_k