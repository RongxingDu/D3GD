import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
import torch

def align_metrics_for_plotting(history):
    lengths = {k: len(v) for k, v in history.items()}
    if not lengths: return pd.DataFrame()
    max_len = max(lengths.values())
    
    data = {}
    for k, v in history.items():
        if len(v) == max_len:
            data[k] = v
    
    df = pd.DataFrame(data)
    
    for k, v in history.items():
        if len(v) < max_len and len(v) > 0:
            indices = np.linspace(0, max_len - 1, len(v)).astype(int)
            s = pd.Series(data=v, index=indices)
            df[k] = s.reindex(df.index).ffill()
            
    return df

def plot_training_results(history, save_dir, title_suffix=""):
    """
    Plots metrics using Epochs as x-axis.
    Assumes metrics in history['test_acc'] are recorded once per epoch.
    """
    if not history:
        print("Warning: No history data to plot.")
        return

    # Helper to get the main metric list
    if 'test_mse' in history and len(history['test_mse']) > 0:
        metric_key = 'test_mse'
        ylabel = 'MSE (Log Scale)'
        is_log = True
    elif 'test_acc' in history and len(history['test_acc']) > 0:
        metric_key = 'test_acc'
        ylabel = 'Accuracy (%)'
        is_log = False
    else:
        return

    # Extract Data
    y_data = np.array(history[metric_key])
    epochs = np.arange(1, len(y_data) + 1) # X-axis: 1, 2, 3...
    
    # --- Plot 1: Convergence vs Epochs ---
    plt.figure(figsize=(10, 6))
    
    if is_log:
        plt.semilogy(epochs, y_data, label='Average', linewidth=2, color='red')
    else:
        plt.plot(epochs, y_data, label='Average', linewidth=2, color='blue')
        
        # Add Min/Max Spread if available
        if 'test_acc_min' in history and 'test_acc_max' in history:
            y_min = np.array(history['test_acc_min'])
            y_max = np.array(history['test_acc_max'])
            
            # Ensure lengths match before plotting spread
            if len(y_min) == len(epochs) and len(y_max) == len(epochs):
                plt.plot(epochs, y_min, linestyle='--', color='red', alpha=0.5, label='Min (Worst)')
                plt.plot(epochs, y_max, linestyle='--', color='green', alpha=0.5, label='Max (Best)')
                plt.fill_between(epochs, y_min, y_max, color='blue', alpha=0.1)

    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.title(f'Convergence vs Epochs {title_suffix}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'convergence_epoch.png'))
    plt.close()
    
    # --- Plot 2: Efficiency (Metric vs Comm Volume) ---
    # Use the NEW aligned key 'test_comm_volume'
    if 'test_comm_volume' in history and len(history['test_comm_volume']) == len(y_data):
        x_comm = np.array(history['test_comm_volume'])
        
        plt.figure(figsize=(10, 6))
        if is_log:
            plt.semilogy(x_comm, y_data, marker='o', markersize=4, label=metric_key, color='purple')
        else:
            plt.plot(x_comm, y_data, marker='o', markersize=4, label=metric_key, color='green')
            
        plt.xlabel('Communication Volume (Floats)')
        plt.ylabel(ylabel)
        plt.title(f'Efficiency: Performance vs Communication {title_suffix}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, 'efficiency_epoch.png'))
        plt.close()

    # --- Plot 3: Consensus Error (Optional, usually logged per step) ---
    if 'consensus_error' in history and len(history['consensus_error']) > 0:
        cons_data = np.array(history['consensus_error'])
        # If consensus is logged per step, use steps. If per epoch, use epochs.
        # Usually logged per epoch in your new setup? 
        # If it's much longer than epochs, we plot vs Steps.
        
        x_axis = np.arange(1, len(cons_data) + 1)
        x_label = 'Steps'
        
        # Heuristic: if lengths match, assume it's per epoch
        if len(cons_data) == len(epochs):
            x_label = 'Epochs'
            
        plt.figure(figsize=(10, 6))
        plt.semilogy(x_axis, cons_data, label='Consensus Error', color='orange')
        plt.xlabel(x_label)
        plt.ylabel('Consensus Error (Frobenius Norm)')
        plt.title(f'Consensus Error {title_suffix}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(save_dir, 'consensus_error.png'))
        plt.close()

    # Append to utils/plotting.py

def plot_topology_heatmap(matrix, save_path, title="Topology Weights"):
    """
    Plots a heatmap of the adjacency matrix.
    matrix: numpy array or torch tensor (NxN)
    """
    # Convert to Numpy
    if hasattr(matrix, 'detach'):
        matrix = matrix.detach().cpu().numpy()
        
    plt.figure(figsize=(12, 10))
    
    # Use a heatmap with a mask for zero values to make them distinct (optional)
    # cmap="viridis" is good for showing intensity (Yellow=High, Purple=Low)
    sns.heatmap(matrix, cmap="viridis", vmin=0.0, center=None, square=True, 
                cbar_kws={'label': 'Edge Weight'})
    
    plt.title(title)
    plt.xlabel("Target Node Index")
    plt.ylabel("Source Node Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Topology Heatmap saved to {save_path}")