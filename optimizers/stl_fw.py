import torch
import numpy as np
import scipy.optimize
from .base import DecentralizedOptimizer

class STL_FW(DecentralizedOptimizer):
    def __init__(self, config, nodes, topology_manager):
        super().__init__(config, nodes, topology_manager)
        
        self.params = config.algorithm.stl_fw
        self.sparsity_budget = self.params.sparsity_budget
        self.lr = config.training.global_lr
        
        # 1. Collect Class Distributions
        self.H = self._collect_label_distributions()
        self.H_centered = self.H - self.H.mean(dim=0, keepdim=True)
        
        # Pre-compute K = H H^T
        self.K = torch.matmul(self.H_centered, self.H_centered.T)
        
        # 2. Run Frank-Wolfe
        print("STL-FW: Starting Topology Learning Phase...")
        self.W = self._run_frank_wolfe()
        print("STL-FW: Topology Learning Complete.")
        
        # 3. Apply logic
        self.topo.update_weights(self.W)
        self.theta_curr = self._collect_node_params()

    def _collect_label_distributions(self):
        dists = []
        num_classes = 10 
        
        for node in self.nodes:
            dataset = node.data_loader.dataset
            targets = None
            
            # --- Robust Target Extraction ---
            if isinstance(dataset, torch.utils.data.Subset):
                underlying = dataset.dataset
                indices = dataset.indices
                if hasattr(underlying, 'targets'):
                    targets = np.array(underlying.targets)[indices]
                elif hasattr(underlying, 'tensors'):
                    targets = underlying.tensors[1][indices].detach().cpu().numpy().flatten()
            elif hasattr(dataset, 'tensors'):
                targets = dataset.tensors[1].detach().cpu().numpy().flatten()
            elif hasattr(dataset, 'targets'):
                targets = np.array(dataset.targets)
                
            if targets is None:
                dists.append(torch.ones(num_classes) / num_classes)
                continue

            # --- Distribution Creation ---
            is_regression = (targets.dtype.kind in 'fc') or (len(np.unique(targets)) > num_classes * 2)
            
            if is_regression:
                counts, _ = np.histogram(targets, bins=num_classes, density=False)
            else:
                counts = np.bincount(targets.astype(int), minlength=num_classes)
            
            total = counts.sum()
            probs = counts / total if total > 0 else np.ones(num_classes) / num_classes
            dists.append(torch.tensor(probs, dtype=torch.float32))
            
        return torch.stack(dists).to(self.device)

    def _run_frank_wolfe(self):
        N = self.num_nodes
        
        # --- FIX: Initialize with Connected Metropolis Weights instead of Identity ---
        # This prevents the "Disconnected Graph" explosion problem.
        W = self.topo.get_weights().clone()
        
        physical_adj = self.topo.physical_mask.cpu().numpy()
        np.fill_diagonal(physical_adj, 1.0)
        
        for k in range(self.sparsity_budget):
            # 1. Gradient
            Grad = torch.matmul(W, self.K)
            
            # 2. Oracle
            cost_matrix = Grad.detach().cpu().numpy()
            cost_matrix[physical_adj == 0] = 1e8 
            
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            S = torch.zeros((N, N), device=self.device)
            S[row_ind, col_ind] = 1.0
            S_sym = (S + S.T) / 2.0
            
            # 3. Line Search
            Delta = S_sym - W
            A = torch.matmul(W, self.H_centered)
            B = torch.matmul(Delta, self.H_centered)
            
            numerator = torch.sum(A * B)
            denominator = torch.sum(B * B)
            
            if denominator < 1e-9:
                gamma = 0.0
            else:
                gamma = -numerator / denominator
                
            gamma = torch.clamp(gamma, 0.0, 1.0)
            
            # 4. Update
            W = (1 - gamma) * W + gamma * S_sym
            
        return W

    def step(self):
        grads = self._collect_gradients()

        # --- [NEW] Gradient Clipping ---
        # Standard clipping for D-SGD
        grad_norm = torch.norm(grads, p=2, dim=1, keepdim=True)
        max_norm = 5.0
        clip_coef = torch.clamp(max_norm / (grad_norm + 1e-6), max=1.0)
        grads = grads * clip_coef
        # -------------------------------
        theta_mixed = self.neighbor_mix(self.theta_curr, self.W)
        
        # Gradient Descent Step
        theta_next = theta_mixed - self.lr * grads
        
        self._distribute_node_params(theta_next)
        self.theta_curr = theta_next
        self.global_step += 1
        
        # Metrics
        self.compute_consensus_error() 
        edges = (torch.count_nonzero(self.W).item() - self.num_nodes)
        vol = edges * self.theta_curr.shape[1]
        self.log_metrics(0.0, vol)