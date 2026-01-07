import torch
import torch.nn as nn
import copy

class Node:
    def __init__(self, node_id, model_template, data_loader, device, loss_fn):
        self.id = node_id
        self.device = device
        
        # 1. Local Model (Deep Copy ensures independence)
        self.model = copy.deepcopy(model_template).to(device)
        
        # 2. Data
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        
        # 3. Loss Function
        self.loss_fn = loss_fn
        
        # 4. Optimization State (Buffers)
        self._param_buffer = None 
        self._grad_buffer = None
        
    def get_batch(self):
        """Infinite iterator over local data."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            batch = next(self.data_iter)
        return [t.to(self.device) for t in batch]

    def compute_gradient(self):
        """
        Runs forward pass and backward pass.
        Returns: Flattened Gradient Vector.
        """
        inputs, targets = self.get_batch()
        
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        
        # Flatten gradients into a single vector
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros_like(param.view(-1)))
                
        self._grad_buffer = torch.cat(grads)
        return self._grad_buffer

    def get_params(self):
        """Returns current parameters as flattened vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.data.view(-1))
        self._param_buffer = torch.cat(params)
        return self._param_buffer

    def set_params(self, new_params_flat):
        """Updates internal model parameters from a flattened vector."""
        curr_idx = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data.copy_(
                new_params_flat[curr_idx : curr_idx + numel].view_as(param)
            )
            curr_idx += numel

    def evaluate(self, test_loader):
        """
        Evaluates the model on the global test set.
        - Returns MSE for Regression (if loss_fn is MSELoss)
        - Returns Accuracy (%) for Classification otherwise
        """
        self.model.eval()
        total_metric = 0.0
        total_samples = 0
        
        is_regression = isinstance(self.loss_fn, nn.MSELoss)
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                
                if is_regression:
                    # For Regression: Metric is MSE Loss
                    # We assume targets are (N, 1) and outputs are (N, 1)
                    batch_mse = self.loss_fn(outputs, targets).item()
                    # Weighted average by batch size
                    total_metric += batch_mse * targets.size(0)
                else:
                    # For Classification: Metric is Accuracy
                    _, predicted = outputs.max(1)
                    correct = predicted.eq(targets).sum().item()
                    total_metric += correct
                    
                total_samples += targets.size(0)
                
        self.model.train()
        
        if total_samples == 0:
            return 0.0
            
        avg_metric = total_metric / total_samples
        
        if not is_regression:
            avg_metric *= 100.0 # Convert to percentage for accuracy
            
        return avg_metric