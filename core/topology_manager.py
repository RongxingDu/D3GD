import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from .network import PhysicalNetwork

class TopologyManager:
    def __init__(self, config):
        self.cfg = config
        self.device = config.device
        
        # 1. Create Physical Layer
        self.physical_layer = PhysicalNetwork(
            config.environment.num_nodes, 
            config.environment.graph_type,
            config.environment.connectivity_radius,
            config.seed
        )
        
        # 2. Immutable Physical Constraints (Binary Mask)
        # Convert to Torch Tensor for GPU operations
        self.physical_mask = torch.tensor(
            self.physical_layer.get_adjacency(), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # 3. Dynamic State (The actual weights used for communication)
        # Initialize with Metropolis-Hastings (Fair Baseline)
        self.current_weights = self._init_metropolis_hastings()

    def _init_metropolis_hastings(self):
        """
        Constructs a doubly stochastic matrix compatible with the physical graph.
        W_ij = 1 / (1 + max(d_i, d_j)) if connected
        """
        adj = self.physical_layer.get_adjacency()
        degrees = np.sum(adj, axis=1)
        W = np.zeros_like(adj)
        
        rows, cols = np.nonzero(adj)
        for i, j in zip(rows, cols):
            W[i, j] = 1.0 / (1.0 + max(degrees[i], degrees[j]))
            
        # Fill diagonal: W_ii = 1 - sum(W_ij)
        for i in range(len(W)):
            W[i, i] = 1.0 - np.sum(W[i, :])
            
        return torch.tensor(W, dtype=torch.float32, device=self.device)

    def get_weights(self):
        """Returns the current weight matrix (NxN) on device."""
        return self.current_weights

    def update_weights(self, new_weights):
        """
        Used by D3GD and STL-FW to update logical topology.
        CRITICAL: Enforces physical constraints.
        """
        # 1. Mask out non-physical edges (Keep Diagonal)
        # We allow self-loops (diagonal) implicitly in logic, 
        # but physical_mask usually has 0 diagonal. 
        # So we mask off-diagonal only.
        
        mask = self.physical_mask + torch.eye(self.cfg.environment.num_nodes, device=self.device)
        constrained_weights = new_weights * (mask > 0).float()
        
        # 2. Update
        self.current_weights = constrained_weights
    
    

    def visualize_topology(self, save_path="topology.png", title="Current Topology"):
        """
        Visualizes the *Logical* weights on top of the *Physical* positions.
        Thicker lines = Higher weight.
        """
        plt.figure(figsize=(10, 8))
        
        # Get physical positions
        pos = self.physical_layer.positions
        
        # Get logical weights (CPU)
        W = self.current_weights.detach().cpu().numpy()
        
        # Draw nodes
        nx.draw_networkx_nodes(self.physical_layer.graph, pos, node_size=100, node_color='skyblue')
        
        # Draw edges based on W
        # We only draw edges where W_ij > 0.01 to avoid clutter
        rows, cols = np.where(W > 0.01)
        edges = zip(rows, cols)
        
        for i, j in edges:
            if i == j: continue # Don't draw self-loops
            
            weight = W[i, j]
            # STL-FW/AC-GT might have 0/1 weights, D3GD has floats
            # Width scaling
            width = 1.0 + (weight * 5.0) 
            
            # Color: Black for strong, Gray for weak
            alpha = min(1.0, weight * 2.0 + 0.1)
            
            # Draw line manually to control width/alpha per edge
            plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 
                     color='black', linewidth=width, alpha=alpha, zorder=1)

        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        print(f"Topology visualization saved to {save_path}")