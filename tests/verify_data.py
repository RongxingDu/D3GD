import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import ExperimentConfig
from data.loaders import DecentralizedDataInterface

def visualize_skew(data_interface, save_path="data_distribution.png"):
    """
    Creates a heatmap: Node ID vs Class Label Count.
    """
    node_map = data_interface.node_indices_map
    dataset = data_interface.train_dataset
    
    # Assuming classification (MNIST/CIFAR)
    if not hasattr(dataset, 'targets'):
        print("Skipping visualization for regression dataset.")
        return

    num_nodes = len(node_map)
    num_classes = 10 # standard for MNIST/CIFAR
    
    # Distribution Matrix [Nodes x Classes]
    heatmap = np.zeros((num_nodes, num_classes))
    
    print("Computing class distributions per node...")
    # Handle different dataset types (Torchvision vs TensorDataset)
    if isinstance(dataset.targets, list):
        labels = np.array(dataset.targets)
    elif isinstance(dataset.targets, torch.Tensor):
        labels = dataset.targets.numpy()
    else:
        labels = dataset.targets
    
    for node_id, indices in node_map.items():
        node_labels = labels[indices]
        counts = np.bincount(node_labels, minlength=num_classes)
        heatmap[node_id] = counts

    # Plotting
    plt.figure(figsize=(12, 8))
    # FIX: Changed 'Viridis' to 'viridis' (lowercase)
    plt.imshow(heatmap.T, aspect='auto', cmap='viridis', interpolation='nearest') 
    plt.colorbar(label='Number of Samples')
    plt.xlabel('Node ID')
    plt.ylabel('Class Label')
    plt.title(f'Data Distribution (Alpha={data_interface.cfg.data.heterogeneity.alpha})')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Verification plot saved to {save_path}")

if __name__ == "__main__":
    import torch # Import torch here to handle the check above
    
    # 1. Load Config (Simulating MNIST Heterogeneous)
    config_path = "configs/cifar_resnet.yaml"
    cfg = ExperimentConfig(config_path)
    
    # 2. Initialize Data Interface
    data_int = DecentralizedDataInterface(cfg)
    
    # 3. Verify Basic Constraints
    total_samples = sum([len(idx) for idx in data_int.node_indices_map.values()])
    print(f"Total partitioned samples: {total_samples}")
    print(f"Original dataset size: {len(data_int.train_dataset)}")
    
    assert total_samples == len(data_int.train_dataset), "Lost data samples during partition!"
    
    # 4. Visual Verification
    visualize_skew(data_int)