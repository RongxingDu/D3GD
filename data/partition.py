import numpy as np
from collections import defaultdict

class DataPartitioner:
    def __init__(self, data, labels, num_nodes, seed=42):
        self.data = data
        self.labels = np.array(labels)
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)
        
    def partition_iid(self):
        """Randomly splits indices into N equal chunks."""
        indices = np.arange(len(self.labels))
        self.rng.shuffle(indices)
        return np.array_split(indices, self.num_nodes)

    def partition_dirichlet(self, alpha=0.1):
        """
        Standard Non-IID partition strategy (Label Skew).
        """
        min_size = 0
        num_classes = len(np.unique(self.labels))
        
        # Retry loop to ensure valid partition (no empty nodes)
        while min_size < 10:
            indices_map = defaultdict(list)
            
            for k in range(num_classes):
                idx_k = np.where(self.labels == k)[0]
                self.rng.shuffle(idx_k)
                
                # 1. Sample Proportions
                proportions = self.rng.dirichlet(np.repeat(alpha, self.num_nodes))
                
                # 2. Calculate integer split points based on proportions
                # The cumsum defines the boundaries for the splits
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                # 3. Split the indices for class k
                node_splits = np.split(idx_k, split_points)
                
                # 4. Assign
                for node_id, batch in enumerate(node_splits):
                    indices_map[node_id].extend(batch)

            min_size = min([len(v) for v in indices_map.values()])
            if min_size < 10:
                print(f"  ... Resampling partition (found empty node)")

        return indices_map

    def partition_manual_clusters(self, num_clusters=2):
        indices = np.arange(len(self.labels))
        return np.array_split(indices, self.num_nodes)