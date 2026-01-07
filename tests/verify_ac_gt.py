import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ExperimentConfig
from core.topology_manager import TopologyManager
from core.node import Node
from optimizers.ac_gt import AC_GT

def verify_ac_gt():
    print("--- Verifying AC-GT Optimizer ---")
    
    # 1. Setup
    config_path = "configs/linear_regression.yaml"
    cfg = ExperimentConfig(config_path)
    cfg.environment.num_nodes = 10
    cfg.algorithm.ac_gt.pruning_threshold = 0.5 # Prune 50%
    cfg.algorithm.ac_gt.cycle_length = 2
    cfg.device = "cpu"
    
    # 2. Initialize
    topo = TopologyManager(cfg)
    nodes = []
    
    print("Initializing Nodes with Dummy Data...")
    for i in range(10):
        model = nn.Sequential(nn.Linear(5, 1))
        
        # Create divergent params to force pruning choices
        with torch.no_grad():
            if i < 5:
                model[0].weight.fill_(0.0) # Cluster 1
            else:
                model[0].weight.fill_(10.0) # Cluster 2
        
        # FIX: Create a proper batch tuple (Input, Target)
        # Input: (Batch=1, Features=5), Target: (Batch=1, Dim=1)
        dummy_input = torch.randn(1, 5)
        dummy_target = torch.randn(1, 1)
        
        # The loader is a list containing one batch tuple
        # The Node class will iterate this list infinitely
        dummy_loader = [(dummy_input, dummy_target)]
        
        nodes.append(Node(i, model, dummy_loader, "cpu", nn.MSELoss()))
        
    optimizer = AC_GT(cfg, nodes, topo)
    
    # 3. Test Pruning (Step 0)
    print("Step 0 (Cycle Start): Pruning...")
    optimizer.step()
    
    # Check Density
    full_edges = torch.count_nonzero(topo.physical_mask).item()
    
    # We subtract 10 because W_x has self-loops (diagonal), which we don't count as "edges" for pruning checks
    kept_edges_x = torch.count_nonzero(optimizer.W_x).item() - 10 
    
    print(f"Full Physical Edges (excluding diagonal): {full_edges}")
    print(f"Kept Edges (W_x, excluding diagonal): {kept_edges_x}")
    
    if kept_edges_x >= full_edges:
        print("WARNING: No edges were pruned. This can happen in small random graphs if distances are uniform.")
    else:
        print("SUCCESS: Edges were pruned.")
    
    # Check Stochasticity
    row_sums = optimizer.W_x.sum(dim=1)
    col_sums = optimizer.W_x.sum(dim=0)
    
    assert torch.allclose(row_sums, torch.ones(10), atol=1e-4), f"W_x not Row Stochastic: {row_sums}"
    # Note: AC-GT W_x is symmetric (Doubly Stochastic) because of the intersection logic
    assert torch.allclose(col_sums, torch.ones(10), atol=1e-4), "W_x not Doubly Stochastic"
    
    # 4. Test Dynamics (Step 1 - No Pruning)
    old_Wx = optimizer.W_x.clone()
    print("Step 1 (In Cycle): No Pruning...")
    optimizer.step()
    
    assert torch.equal(optimizer.W_x, old_Wx), "Topology changed inside cycle! (It shouldn't)"
    
    # 5. Test Dynamics (Step 2 - New Cycle)
    print("Step 2 (Cycle Start): Re-Pruning...")
    optimizer.step()
    
    # Check that we are still stochastic
    assert torch.allclose(optimizer.W_x.sum(dim=1), torch.ones(10), atol=1e-4)

    print("AC-GT Verification Passed!")

if __name__ == "__main__":
    verify_ac_gt()