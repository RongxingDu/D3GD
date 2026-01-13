import torch
import torch.nn as nn
import numpy as np
import random
import os
import argparse
from tqdm import tqdm

from utils.config import ExperimentConfig, parse_args
from utils.logger import ExperimentLogger
from utils.plotting import plot_training_results
from core.network import PhysicalNetwork
from core.topology_manager import TopologyManager
from core.node import Node
from core.models import get_model
from data.loaders import DecentralizedDataInterface
from optimizers.di_dgd import Di_DGD
from optimizers.d3gd import D3GD
from optimizers.stl_fw import STL_FW
from optimizers.ac_gt import AC_GT
from utils.plotting import plot_training_results, plot_topology_heatmap 

def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer_class(algo_name):
    if algo_name == 'di_dgd': return Di_DGD
    if algo_name == 'd3gd': return D3GD
    if algo_name == 'stl_fw': return STL_FW
    if algo_name == 'ac_gt': return AC_GT
    raise ValueError(f"Unknown algorithm: {algo_name}")

def get_effective_topology(optimizer):
    """
    Extracts the effective adjacency matrix used for communication.
    """
    if hasattr(optimizer, 'A_bar'): # D3GD
        # A_effective = (1 - delta) * A_bar + delta * A0
        return (1 - optimizer.delta) * optimizer.A_bar + optimizer.delta * optimizer.A0
    elif hasattr(optimizer, 'A'): # Di-DGD
        return optimizer.A
    elif hasattr(optimizer, 'topo'): 
        # AC-GT and STL-FW usually update the TopologyManager directly
        return optimizer.topo.get_weights()
    return None

def main():
    args = parse_args()
    cfg = ExperimentConfig(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed)
    
    print(f"=== Starting Experiment: {cfg.experiment_name} ===")
    print(f"Algorithm: {args.algo.upper()}")
    
    logger = ExperimentLogger(cfg, args.algo)
    
    print("Initializing Data...")
    data_interface = DecentralizedDataInterface(cfg)
    test_loader = data_interface.get_test_dataloader()
    
    print("Initializing Topology...")
    topo_manager = TopologyManager(cfg)
    
    # --- VISUALIZATION 1: INITIAL TOPOLOGY ---
    initial_weights = topo_manager.get_weights()
    topo_manager.visualize_topology(
        save_path=os.path.join(logger.save_dir, "topology_graph_initial.png"),
        title="Initial Topology (Graph)"
    )
    plot_topology_heatmap(
        initial_weights, 
        save_path=os.path.join(logger.save_dir, "topology_heatmap_initial.png"),
        title="Initial Edge Weights (Heatmap)"
    )
    # -----------------------------------------

    print(f"Initializing {cfg.environment.num_nodes} Nodes...")
    
    nodes = []
    
    # Define Loss Function
    if cfg.training.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif cfg.training.loss == 'cross_entropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unknown loss function")
    
    # Input/Output Dims
    input_dim = None
    if 'synthetic' in cfg.data.dataset:
        input_dim = cfg.data.input_dim
    
    output_dim = 1
    if hasattr(cfg.training, 'output_dim'):
        output_dim = cfg.training.output_dim
    else:
        assert print("Output dimension must be specified in config for datasets.")

    for i in range(cfg.environment.num_nodes):
        model = get_model(cfg.training.model, input_dim, output_dim)
        
        # Adjust Linear Model for Output Dim (e.g. 2 classes)
        if cfg.training.model == 'linear' and output_dim != 1:
            # Re-initialize the linear layer
            model.linear = nn.Linear(input_dim, output_dim)
            
        node_loader = data_interface.get_node_dataloader(i)
        
        node = Node(
            node_id=i,
            model_template=model,
            data_loader=node_loader,
            device=cfg.device,
            loss_fn=loss_fn
        )
        nodes.append(node)

    print("\n=== Verifying Data Heterogeneity ===")
    for i, node in enumerate(nodes):
        try:
            # 1. Access the dataset (unwrap Subset if present)
            ds = node.data_loader.dataset
            indices = None
            
            if hasattr(ds, 'indices'):
                indices = ds.indices
                ds = ds.dataset # Unwrap to get the underlying full dataset
            
            # 2. Extract full targets based on dataset type
            if hasattr(ds, 'targets'):
                # Standard datasets like MNIST/CIFAR
                full_targets = ds.targets
                if isinstance(full_targets, list):
                    full_targets = torch.tensor(full_targets)
            elif hasattr(ds, 'tensors'):
                # TensorDataset (Synthetic data)
                full_targets = ds.tensors[1] # usually [features, labels]
            else:
                raise AttributeError("Dataset has no 'targets' or 'tensors'")
                
            # 3. Filter targets for this specific node
            if indices is not None:
                local_targets = full_targets[indices]
            else:
                local_targets = full_targets
                
            # 4. Count and Print
            # Ensure it's a flat tensor for counting
            if local_targets.dim() > 1:
                local_targets = local_targets.argmax(dim=1) if local_targets.shape[1] > 1 else local_targets.flatten()
                
            unique, counts = torch.unique(local_targets.long(), return_counts=True)
            dist = dict(zip(unique.tolist(), counts.tolist()))
            print(f"Node {i}: {dist}")
            
        except Exception as e:
            print(f"Node {i}: Could not inspect data - {e}")
    print("====================================\n")
        
    print("Initializing Optimizer...")
    OptimizerClass = get_optimizer_class(args.algo)
    optimizer = OptimizerClass(cfg, nodes, topo_manager)
    
    # print("Starting Training Loop...")
    # if hasattr(cfg.training, 'max_iterations'):
    #     total_steps = cfg.training.max_iterations
    # else:
    #     if len(data_interface.train_dataset) > 0:
    #         steps_per_epoch = len(data_interface.train_dataset) // (cfg.environment.num_nodes * cfg.training.batch_size)
    #         total_steps = cfg.training.max_epochs * steps_per_epoch
    #     else:
    #         total_steps = 100
            
    # if hasattr(cfg.training, 'eval_interval'):
    #     eval_interval = cfg.training.eval_interval
    # else:
    #     eval_interval = total_steps // 20 if total_steps > 20 else 1
    

    # 1. Setup Data Loaders
    # Use the small subset for the training loop checks
    val_loader = data_interface.get_val_dataloader(size=1000) 
    # Keep the full test set for the final evaluation
    full_test_loader = data_interface.get_test_dataloader()

    print("Starting Training Loop...")
    
    # 2. Calculate Steps Per Epoch
    if len(data_interface.train_dataset) > 0:
        # Global Epoch = Total Data / (Nodes * Batch Size)
        steps_per_epoch = len(data_interface.train_dataset) // (cfg.environment.num_nodes * cfg.training.batch_size)
        if steps_per_epoch < 1: steps_per_epoch = 1
    else:
        steps_per_epoch = 1

    # 3. Set Interval to 1 Epoch
    eval_interval = steps_per_epoch
    print(f"  -> Training for {cfg.training.max_epochs} epochs")
    print(f"  -> Steps per Epoch: {steps_per_epoch}")
    print(f"  -> Evaluation Interval: {eval_interval} steps")

    # Recalculate total steps if needed (ensure it covers max_epochs)
    total_steps = cfg.training.max_epochs * steps_per_epoch
    
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        optimizer.step()
        
        # 4. Evaluate using Val Loader (Fast)
        if (step + 1) % eval_interval == 0:
            # Pass the small val_loader here
            avg_metric = optimizer.evaluate(val_loader)
            
            cons_err = optimizer.history['consensus_error'][-1] if optimizer.history['consensus_error'] else 0.0
            metric_name = "MSE" if isinstance(loss_fn, nn.MSELoss) else "Acc"
            
            pbar.set_postfix({
                'Epoch': f"{(step + 1) // steps_per_epoch}",
                metric_name: f"{avg_metric:.2f}", 
                'ConsErr': f"{cons_err:.4f}"
            })
            
    print("Training Complete.")
    
    # 5. Final Evaluation on FULL Test Set
    print("Running Final Evaluation on Full Test Set...")
    final_acc = optimizer.evaluate(full_test_loader)
    print(f"Final Test Accuracy: {final_acc:.2f}%")
    print("Saving metrics...")
    
    logger.save_history(optimizer.history)
    plot_training_results(optimizer.history, logger.save_dir, title_suffix=f"({args.algo})")
    
    if args.algo in ['d3gd', 'stl_fw']:
        if hasattr(optimizer, 'W'):
             topo_manager.current_weights = optimizer.W
        elif hasattr(optimizer, 'A_bar'):
             topo_manager.current_weights = optimizer.A_bar
             
        topo_manager.visualize_topology(
            save_path=os.path.join(logger.save_dir, "topology_final.png"),
            title=f"Final Learned Topology ({args.algo})"
        )
    # --- VISUALIZATION 2: FINAL TOPOLOGY ---
    final_weights = get_effective_topology(optimizer)
    
    if final_weights is not None:
        # Plot Heatmap
        plot_topology_heatmap(
            final_weights, 
            save_path=os.path.join(logger.save_dir, "topology_heatmap_final.png"),
            title=f"Final Learned Topology ({args.algo.upper()})"
        )
        
        # Plot Graph (Optional: Update topo manager to visualize final graph structure)
        # We temporarily set the manager's weights to the final ones to use its plotting tool
        original_weights = topo_manager.current_weights
        topo_manager.update_weights(final_weights) 
        topo_manager.visualize_topology(
            save_path=os.path.join(logger.save_dir, "topology_graph_final.png"),
            title=f"Final Topology Graph ({args.algo.upper()})"
        )
        # Restore (though program is ending, it's good practice)
        topo_manager.current_weights = original_weights

if __name__ == "__main__":
    main()