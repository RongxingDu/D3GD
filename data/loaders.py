import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from .partition import DataPartitioner

class DecentralizedDataInterface:
    def __init__(self, config):
        self.cfg = config
        self.train_dataset = None
        self.test_dataset = None
        self.node_indices_map = {} 
        
        self._prepare_data()
        self._partition_data()

    def _prepare_data(self):
        name = self.cfg.data.dataset
        root = './dataset_storage'
        
        if name == 'synthetic_linear':
            self._generate_synthetic_linear()
        elif name == 'synthetic_binary': # <--- NEW BLOCK
            self._generate_synthetic_binary()
        elif name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
            self.test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
        elif name == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _generate_synthetic_linear(self):
        # ... (Same as before) ...
        # Copied for completeness
        N = self.cfg.environment.num_nodes * 100 
        d = self.cfg.data.input_dim
        true_w = torch.randn(d, 1)
        X_list, y_list = [], []
        num_clusters = self.cfg.data.heterogeneity.num_clusters
        points_per = N // num_clusters
        
        for c in range(num_clusters):
            cluster_mean = torch.randn(d) * 2 
            X_c = torch.randn(points_per, d) + cluster_mean
            noise = torch.randn(points_per, 1) * self.cfg.data.noise_level
            y_c = X_c @ true_w + noise
            X_list.append(X_c)
            y_list.append(y_c)
        self.train_dataset = TensorDataset(torch.cat(X_list), torch.cat(y_list))
        self.test_dataset = self.train_dataset 

    def _generate_synthetic_binary(self):
        """
        Generates Binary Classification Data.
        y = 1 if (Xw + b + noise) > 0 else 0
        """
        N = self.cfg.environment.num_nodes * self.cfg.data.num_samples_per_node
        d = self.cfg.data.input_dim
        
        # Ground Truth Hyperplane
        true_w = torch.randn(d, 1)
        
        X_list, y_list = [], []
        num_clusters = self.cfg.data.heterogeneity.num_clusters
        points_per = N // num_clusters
        
        print(f"Generating Synthetic Binary Data ({N} samples, {d} dim, {num_clusters} clusters)...")
        
        for c in range(num_clusters):
            # Cluster bias to induce non-IID feature distribution
            cluster_mean = torch.randn(d) * 2 
            X_c = torch.randn(points_per, d) + cluster_mean
            
            # Linear combination
            logits = X_c @ true_w + (torch.randn(points_per, 1) * 0.5) # Add noise
            
            # Threshold to get classes 0 and 1
            y_c = (logits > 0).long() # LongTensor for CrossEntropy
            
            X_list.append(X_c)
            y_list.append(y_c.squeeze()) # Flatten to (N,)
            
        # Create TensorDataset
        self.train_dataset = TensorDataset(torch.cat(X_list), torch.cat(y_list))
        self.test_dataset = self.train_dataset 

    def _partition_data(self):
        targets = []
        if hasattr(self.train_dataset, 'targets'):
            targets = self.train_dataset.targets
        elif hasattr(self.train_dataset, 'tensors'): 
            # For synthetic binary, the second tensor IS the targets (0s and 1s)
            # For synthetic linear, it was floats. 
            # We convert to numpy for the Partitioner.
            targets = self.train_dataset.tensors[1].numpy()
            if targets.ndim > 1: targets = targets.flatten()
            
        partitioner = DataPartitioner(
            self.train_dataset, 
            targets, 
            self.cfg.environment.num_nodes,
            seed=self.cfg.seed
        )
        
        method = self.cfg.data.heterogeneity.type
        if method == 'dirichlet':
            alpha = self.cfg.data.heterogeneity.alpha
            print(f"Partitioning Data: Dirichlet (Alpha={alpha})...")
            self.node_indices_map = partitioner.partition_dirichlet(alpha)
        elif method == 'manual_clusters':
            print("Partitioning Data: Manual Clusters...")
            self.node_indices_map = partitioner.partition_manual_clusters()
        elif method == 'iid':
            self.node_indices_map = partitioner.partition_iid()
        else:
            raise ValueError("Unknown partition method")

    def get_node_dataloader(self, node_id):
        indices = self.node_indices_map[node_id]
        subset = Subset(self.train_dataset, indices)
        return DataLoader(
            subset, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=True
        )

    def get_test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1000, shuffle=False)