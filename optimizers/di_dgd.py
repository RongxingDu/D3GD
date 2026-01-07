import torch
from .base import DecentralizedOptimizer

class Di_DGD(DecentralizedOptimizer):
    def __init__(self, config, nodes, topology_manager):
        super().__init__(config, nodes, topology_manager)
        
        # Initialize Y vector (Eigenvector tracker)
        # Shape: (N, 1) - One scalar per node
        self.y = torch.ones(self.num_nodes, 1, device=self.device)
        
        # Pre-compute static A and pi for metrics
        self.A = self.topo.get_weights()
        self.pi = self.topo.get_perron_vector(self.A)
        
        self.lr = config.training.global_lr

    def step(self):
        # 1. Local Compute: Gradients
        # Returns (N, d) tensor
        grads = self._collect_gradients()
        
        # 2. Local Compute: Current Parameters
        theta = self._collect_node_params()
        
        # 3. Communication Phase
        # Theta mixes with A (Row Stochastic): Sum(A_ij * theta_j)
        theta_mixed = self.neighbor_mix(theta, self.A)
        
        # Y mixes with A_Transpose (Column Mixing): Sum(A_ji * y_j)
        y_mixed = self.neighbor_mix(self.y, self.A.T)
        
        # 4. Update Step
        # theta_new = A*theta - lr * grad / (y * n)
        # Note: y is (N, 1), grads is (N, d). Broadcast division.
        
        # Avoid division by zero stability check
        y_denom = y_mixed * self.num_nodes
        gradient_correction = grads / (y_denom + 1e-8)
        
        theta_new = theta_mixed - self.lr * gradient_correction
        
        # 5. Apply Updates
        self._distribute_node_params(theta_new)
        self.y = y_mixed
        
        # 6. Logging
        self.global_step += 1
        
        # Compute Weighted Consensus Error
        # For static Di-DGD, pi is constant
        self.compute_consensus_error(perron_vector=self.pi)
        
        # Track Communications (Non-zeros in A + Non-zeros in A.T)
        # Since A and A.T have same sparsity, volume is 2 * Edges * (ModelSize + 1)
        # +1 for scalar y
        edges = torch.count_nonzero(self.A).item()
        model_size = theta.shape[1]
        vol = edges * (model_size + 1)
        self.log_metrics(0.0, vol) # Loss is calculated separately if needed

    def _collect_gradients(self):
        stack = []
        for node in self.nodes:
            stack.append(node.compute_gradient())
        return torch.stack(stack)