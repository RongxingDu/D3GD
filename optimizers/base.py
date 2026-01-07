import torch
import numpy as np
from abc import ABC, abstractmethod

class DecentralizedOptimizer(ABC):
    def __init__(self, config, nodes, topology_manager):
        self.cfg = config
        self.nodes = nodes
        self.topo = topology_manager
        self.device = config.device
        self.num_nodes = len(nodes)
        self.global_step = 0
        
        # Metrics History
        self.history = {
            "train_loss": [],
            "test_acc": [],     # Keep for compatibility
            "test_mse": [],     # NEW: For regression
            "consensus_error": [],
            "comm_volume": []
        }

    @abstractmethod
    def step(self):
        pass

    def _collect_node_params(self):
        stack = []
        for node in self.nodes:
            stack.append(node.get_params())
        return torch.stack(stack)

    def _distribute_node_params(self, params_tensor):
        for i, node in enumerate(self.nodes):
            node.set_params(params_tensor[i])

    def _collect_gradients(self):
        stack = []
        for node in self.nodes:
            stack.append(node.compute_gradient())
        return torch.stack(stack)

    def neighbor_mix(self, data_tensor, mixing_matrix):
        if mixing_matrix.device != self.device:
            mixing_matrix = mixing_matrix.to(self.device)
        return torch.matmul(mixing_matrix, data_tensor)

    def evaluate(self, test_loader):
        """
        Evaluates model. 
        Detects if metric is Accuracy (Classification) or MSE (Regression).
        """
        metrics = []
        for node in self.nodes:
            # node.evaluate returns MSE (if regression) or Acc% (if classification)
            metrics.append(node.evaluate(test_loader))
        
        avg_metric = sum(metrics) / len(metrics)
        
        # Heuristic: If metric is > 100 (unlikely for acc) or purely small float, check loss type
        # Robust way: Check the loss function of the first node
        is_regression = isinstance(self.nodes[0].loss_fn, torch.nn.MSELoss)
        
        if is_regression:
            self.history['test_mse'].append(avg_metric)
        else:
            self.history['test_acc'].append(avg_metric)
            
        return avg_metric

    def compute_consensus_error(self, perron_vector=None):
        params = self._collect_node_params()
        if perron_vector is not None:
            center = torch.matmul(perron_vector.unsqueeze(0), params)
        else:
            center = params.mean(dim=0, keepdim=True)
            
        diff = params - center
        error = torch.norm(diff, p='fro') ** 2 / self.num_nodes
        self.history['consensus_error'].append(error.item())
        return error.item()

    def log_metrics(self, loss_val, comm_vol):
        self.history['train_loss'].append(loss_val)
        current_total = self.history['comm_volume'][-1] if self.history['comm_volume'] else 0
        self.history['comm_volume'].append(current_total + comm_vol)