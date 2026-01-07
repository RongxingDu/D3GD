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
        
        # 1. Collect Class Distributions (The "Heterogeneity Data")
        # shape: (Num_Nodes, Num_Classes)
        self.H = self._collect_label_distributions()
        
        # Center the H matrix (We want to minimize variance from mean)
        # H_centered = H - mean(H)
        self.H_centered = self.H - self.H.mean(dim=0, keepdim=True)
        
        # Pre-compute covariance K = H H^T for gradient calc
        # Grad f(W) = W * K
        self.K = torch.matmul(self.H_centered, self.H_centered.T)
        
        # 2. Run Frank-Wolfe to learn W
        print("STL-FW: Starting Topology Learning Phase...")
        self.W = self._run_frank_wolfe()
        print("STL-FW: Topology Learning Complete.")
        
        # 3. Apply the learned topology
        self.topo.update_weights(self.W)
        
        # Standard Di-DGD state
        self.theta_curr = self._collect_node_params()

    def _collect_label_distributions(self):
        """
        Aggregates label counts/proportions from all nodes.
        Handles both Classification (Integer Targets) and Regression (Float Targets).
        """
        dists = []
        num_classes = 10 # Default bins/classes
        
        for node in self.nodes:
            dataset = node.data_loader.dataset
            targets = None
            
            # --- Extract Targets Robustly ---
            # Case 1: Subset (Common in partitioning)
            if isinstance(dataset, torch.utils.data.Subset):
                underlying = dataset.dataset
                indices = dataset.indices
                
                if hasattr(underlying, 'targets'):
                    # Torchvision style
                    targets = np.array(underlying.targets)[indices]
                elif hasattr(underlying, 'tensors'):
                    # TensorDataset (Linear Regression)
                    # We assume (X, y) structure, so index 1 is targets
                    targets = underlying.tensors[1][indices].detach().cpu().numpy().flatten()
            
            # Case 2: Direct TensorDataset
            elif hasattr(dataset, 'tensors'):
                targets = dataset.tensors[1].detach().cpu().numpy().flatten()
                
            # Case 3: Direct Torchvision Dataset
            elif hasattr(dataset, 'targets'):
                targets = np.array(dataset.targets)
                
            # Fallback if extraction failed
            if targets is None:
                print(f"Warning: Could not extract targets for node {node.id}. Using uniform dist.")
                dists.append(torch.ones(num_classes) / num_classes)
                continue

            # --- Create Distribution ---
            # Check if Regression (Floats) or Classification (Ints)
            is_regression = (targets.dtype.kind in 'fc') or (len(np.unique(targets)) > num_classes * 2)
            
            if is_regression:
                # REGRESSION: Bin the continuous values into a histogram
                # This allows STL-FW to group nodes with similar output ranges
                counts, _ = np.histogram(targets, bins=num_classes, density=False)
            else:
                # CLASSIFICATION: Standard bincount
                counts = np.bincount(targets.astype(int), minlength=num_classes)
            
            # Normalize to Probability Distribution
            total = counts.sum()
            if total > 0:
                probs = counts / total
            else:
                probs = np.ones(num_classes) / num_classes
                
            dists.append(torch.tensor(probs, dtype=torch.float32))
            
        return torch.stack(dists).to(self.device)

    def _run_frank_wolfe(self):
        """
        Solves: min_W || W * H ||^2 s.t. W is Doubly Stochastic & Physical
        """
        N = self.num_nodes
        # Initialize W as Identity
        W = torch.eye(N, device=self.device)
        
        # Physical mask (Infinity where no edge exists)
        # Used for Hungarian algorithm cost
        physical_adj = self.topo.physical_mask.cpu().numpy()
        np.fill_diagonal(physical_adj, 1.0)
        
        # Optimization Loop
        for k in range(self.sparsity_budget):
            # 1. Compute Gradient
            # Grad J(W) = W * K
            Grad = torch.matmul(W, self.K)
            
            # 2. Linear Oracle (Hungarian Algorithm)
            cost_matrix = Grad.detach().cpu().numpy()
            # Set non-physical edges to infinity cost
            cost_matrix[physical_adj == 0] = 1e8 
            
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
            
            # Construct Permutation Matrix S
            S = torch.zeros((N, N), device=self.device)
            S[row_ind, col_ind] = 1.0
            
            # Symmetrize S
            S_sym = (S + S.T) / 2.0
            
            # 3. Line Search (Exact for Quadratic)
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
        # Standard D-SGD Step with the learned static W
        grads = self._collect_gradients()
        theta_mixed = self.neighbor_mix(self.theta_curr, self.W)
        theta_next = theta_mixed - self.lr * grads
        
        self._distribute_node_params(theta_next)
        self.theta_curr = theta_next
        self.global_step += 1
        
        # Metrics
        self.compute_consensus_error() 
        edges = (torch.count_nonzero(self.W).item() - self.num_nodes)
        vol = edges * self.theta_curr.shape[1]
        self.log_metrics(0.0, vol)