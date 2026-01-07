import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ExperimentConfig
from core.topology_manager import TopologyManager
from core.node import Node
from optimizers.stl_fw import STL_FW

def verify_stl_fw():
    print("--- Verifying STL-FW Optimizer ---")
    
    # 1. Setup
    config_path = "configs/linear_regression.yaml"
    cfg = ExperimentConfig(config_path)
    cfg.environment.num_nodes = 10
    cfg.algorithm.stl_fw.sparsity_budget = 5
    cfg.device = "cpu"
    
    # 2. Setup Nodes with Heterogeneous "Labels"
    nodes = []
    print("Creating Heterogeneous Nodes...")
    for i in range(10):
        # Create a mock dataset class that looks like MNIST/CIFAR
        class MockDataset:
            def __init__(self, node_id):
                # Node 0-4: Mostly Class 0
                # Node 5-9: Mostly Class 1
                if node_id < 5:
                    self.targets = [0] * 90 + [1] * 10
                else:
                    self.targets = [0] * 10 + [1] * 90
            
            def __len__(self):
                return 100
            
            def __getitem__(self, idx):
                # Return dummy tensors compatible with model (2 inputs, 1 target)
                return torch.randn(2), torch.tensor([0.0])

        # Instantiate Dataset
        dataset = MockDataset(i)
        
        # Instantiate Loader (This automatically sets loader.dataset = dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=10)
        
        # --- FIX: REMOVED THE LINE 'loader.dataset = ...' ---
        
        # Patch __iter__ to return a properly batched tensor for the simple step() test
        # We perform this patch because our MockDataset.__getitem__ returns 1D tensors,
        # but the optimizer expects a batch dimension.
        loader.__iter__ = lambda: iter([(torch.randn(10, 2), torch.randn(10, 1))])
        
        # Dummy model
        model = nn.Sequential(nn.Linear(2, 1))
        
        nodes.append(Node(i, model, loader, "cpu", nn.MSELoss()))
        
    topo = TopologyManager(cfg)
    
    # 3. Run STL-FW (Initialization triggers FW)
    print("Initializing STL-FW (Runs Pre-training)...")
    optimizer = STL_FW(cfg, nodes, topo)
    
    # 4. Verify W Properties
    W = optimizer.W
    print("\nLearned W (First 5x5):\n", W[:5, :5].numpy())
    
    # Symmetry
    assert torch.allclose(W, W.T, atol=1e-4), "W is not symmetric"
    print("Check: W is Symmetric.")
    
    # Stochasticity
    row_sums = W.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(10), atol=1e-4), f"W not Row Stochastic: {row_sums}"
    print("Check: W is Doubly Stochastic.")
    
    # Heterogeneity Minimization Check
    # We check if the algorithm assigned weights between Group A (0-4) and Group B (5-9)
    group_a = range(5)
    group_b = range(5, 10)
    
    # We use np.ix_ to slice the tensor
    inter_group_weight = W[np.ix_(group_a, group_b)].mean().item()
    intra_group_weight = W[np.ix_(group_a, group_a)].mean().item()
    
    print(f"Intra-Cluster Weight (Same Data): {intra_group_weight:.4f}")
    print(f"Inter-Cluster Weight (Diff Data): {inter_group_weight:.4f}")
    
    # 5. Run Step
    optimizer.step()
    print("Step execution successful.")
    
    print("STL-FW Verification Passed!")

if __name__ == "__main__":
    verify_stl_fw()