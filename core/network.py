import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

class PhysicalNetwork:
    def __init__(self, num_nodes, graph_type='rgg', radius=0.4, seed=42):
        self.num_nodes = num_nodes
        self.seed = seed
        self.graph = None
        self.positions = None # (x, y) coordinates for visualization
        self.adj_matrix = None # Binary numpy array (physical connectivity)

        self._generate_graph(graph_type, radius)

    def _generate_graph(self, graph_type, radius):
        np.random.seed(self.seed)
        
        if graph_type == 'rgg':
            # Random Geometric Graph: Nodes connected if distance < radius
            # We generate it manually to ensure we have exact coordinates
            pos = np.random.rand(self.num_nodes, 2)
            dist_matrix = cdist(pos, pos)
            
            # Create adjacency: 1 if dist < radius, 0 otherwise (and 0 diagonal)
            adj = (dist_matrix < radius).astype(float)
            np.fill_diagonal(adj, 0)
            
            # Ensure connectivity (if not connected, regenerate)
            # Simple check: NetworkX check
            G = nx.from_numpy_array(adj)
            if not nx.is_connected(G):
                print("Warning: Generated RGG is disconnected. Increasing radius slightly...")
                return self._generate_graph(graph_type, radius + 0.05)
            
            self.positions = pos
            self.adj_matrix = adj
            self.graph = G
            
        elif graph_type == 'ring':
            G = nx.cycle_graph(self.num_nodes)
            self.adj_matrix = nx.to_numpy_array(G)
            self.positions = nx.circular_layout(G)
            self.graph = G
            
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
            
    def get_adjacency(self):
        return self.adj_matrix.copy()