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
    topo_manager.visualize_topology(
        save_path=os.path.join(logger.save_dir, "topology_initial.png"),
        title="Physical Graph"
    )
    
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
    elif cfg.data.dataset in ['mnist', 'cifar10']:
        output_dim = 10

    for i in range(cfg.environment.num_nodes):
        model = get_model(cfg.training.model, input_dim)
        
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
        
    print("Initializing Optimizer...")
    OptimizerClass = get_optimizer_class(args.algo)
    optimizer = OptimizerClass(cfg, nodes, topo_manager)
    
    print("Starting Training Loop...")
    if hasattr(cfg.training, 'max_iterations'):
        total_steps = cfg.training.max_iterations
    else:
        if len(data_interface.train_dataset) > 0:
            steps_per_epoch = len(data_interface.train_dataset) // (cfg.environment.num_nodes * cfg.training.batch_size)
            total_steps = cfg.training.max_epochs * steps_per_epoch
        else:
            total_steps = 100
            
    if hasattr(cfg.training, 'eval_interval'):
        eval_interval = cfg.training.eval_interval
    else:
        eval_interval = total_steps // 20 if total_steps > 20 else 1
    
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:
        optimizer.step()
        
        if (step + 1) % eval_interval == 0 or (step + 1) == total_steps:
            avg_metric = optimizer.evaluate(test_loader)
            cons_err = optimizer.history['consensus_error'][-1] if optimizer.history['consensus_error'] else 0.0
            
            # Label the metric correctly in progress bar
            metric_name = "MSE" if isinstance(loss_fn, nn.MSELoss) else "Acc"
            pbar.set_postfix({
                metric_name: f"{avg_metric:.4f}", 
                'ConsErr': f"{cons_err:.4f}"
            })
            
    print("Training Complete.")
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

if __name__ == "__main__":
    main()