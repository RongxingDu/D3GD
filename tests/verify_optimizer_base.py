import sys
import os
import torch
import torch.nn as nn

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ExperimentConfig
from core.topology_manager import TopologyManager
from core.node import Node
from optimizers.base import DecentralizedOptimizer

# Create a concrete implementation for testing
class TestOptimizer(DecentralizedOptimizer):
    def step(self):
        pass 

def verify_optimizer_base():
    print("--- Verifying Base Optimizer Logic ---")
    
    # 1. Setup Mock Environment
    config_path = "configs/linear_regression.yaml" 
    # Use generic linear config just to initialize objects
    if not os.path.exists(config_path):
        # Create dummy config file if it doesn't exist for this test
        print("Config not found, skipping full integration test, running unit test only.")
    
    cfg = ExperimentConfig(config_path)
    cfg.environment.num_nodes = 3 # Small graph for manual check
    cfg.device = "cpu"
    
    # 2. Setup Nodes with Explicit Values
    # Node 0 params: [1.0, 1.0]
    # Node 1 params: [2.0, 2.0]
    # Node 2 params: [3.0, 3.0]
    nodes = []
    for i in range(3):
        model = nn.Sequential(nn.Linear(2, 1, bias=False)) # 2 params
        with torch.no_grad():
            model[0].weight.fill_(float(i + 1))
        
        # Mock loader/loss
        node = Node(i, model, [torch.randn(1)], cfg.device, nn.MSELoss())
        nodes.append(node)

    # 3. Setup Topology (Manual W)
    # 0 <-> 1 (Weight 0.5), 2 is isolated (Self loop 1.0)
    # W = [[0.5, 0.5, 0.0],
    #      [0.5, 0.5, 0.0],
    #      [0.0, 0.0, 1.0]]
    topo = TopologyManager(cfg)
    W = torch.tensor([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    optimizer = TestOptimizer(cfg, nodes, topo)
    
    # 4. Test Collect
    params = optimizer._collect_node_params()
    print(f"Collected Params:\n{params}")
    # Expected: [[1,1], [2,2], [3,3]]
    assert torch.allclose(params[0], torch.tensor([1.0, 1.0])), "Collect Node 0 failed"
    assert torch.allclose(params[2], torch.tensor([3.0, 3.0])), "Collect Node 2 failed"
    
    # 5. Test Neighbor Mix
    # Expected Result after W * X:
    # Node 0: 0.5*1 + 0.5*2 = 1.5
    # Node 1: 0.5*1 + 0.5*2 = 1.5
    # Node 2: 1.0*3 = 3.0
    mixed = optimizer.neighbor_mix(params, W)
    print(f"Mixed Params:\n{mixed}")
    
    assert torch.allclose(mixed[0], torch.tensor([1.5, 1.5])), "Mixing logic failed for Node 0"
    assert torch.allclose(mixed[2], torch.tensor([3.0, 3.0])), "Mixing logic failed for Node 2"
    
    # 6. Test Distribute
    optimizer._distribute_node_params(mixed)
    new_p0 = nodes[0].get_params()
    assert torch.allclose(new_p0, torch.tensor([1.5, 1.5])), "Distribution failed"

    print("Base Optimizer Verification Passed!")

if __name__ == "__main__":
    verify_optimizer_base()