import sys
import os
import torch
import torch.nn as nn

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ExperimentConfig
from core.topology_manager import TopologyManager
from core.node import Node

def verify_core():
    # 1. Load Config
    config_path = "configs/mnist_lenet.yaml"
    cfg = ExperimentConfig(config_path)
    
    # Force CPU for testing if CUDA not available
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    
    print(f"Testing Core on {cfg.device}...")

    # 2. Test Topology Manager
    print("Initializing TopologyManager...")
    topo = TopologyManager(cfg)
    
    # Check 1: Physical constraints
    adj = topo.physical_layer.get_adjacency()
    print(f"Physical Graph Generated: {topo.physical_layer.num_nodes} nodes.")
    print(f"Average Degree: {adj.sum(axis=1).mean():.2f}")
    
    # Check 2: Metropolis-Hastings Init
    weights = topo.get_weights()
    row_sums = weights.sum(dim=1)
    print(f"Weight Row Sums (First 5): {row_sums[:5].cpu().numpy()}")
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "Weights not Row-Stochastic!"
    
    # 3. Test Visualization
    print("Generating Visualization...")
    topo.visualize_topology(save_path="test_topology_initial.png", title="Initial MH Topology")
    
    # 4. Test Node Operations
    print("Initializing Dummy Node...")
    # Simple Linear Model for testing
    model = nn.Sequential(nn.Linear(10, 1))
    
    # Dummy Data Loader
    dummy_input = torch.randn(10, 10)
    dummy_target = torch.randn(10, 1)
    dummy_dataset = torch.utils.data.TensorDataset(dummy_input, dummy_target)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=2)
    
    node = Node(
        node_id=0, 
        model_template=model, 
        data_loader=dummy_loader, 
        device=cfg.device, 
        loss_fn=nn.MSELoss()
    )
    
    # Check Gradient Computation
    grad = node.compute_gradient()
    print(f"Gradient Computed. Shape: {grad.shape}")
    assert grad.shape[0] == 11, "Gradient shape mismatch (10 weights + 1 bias)"
    
    print("Core Module Verification Passed!")

if __name__ == "__main__":
    verify_core()