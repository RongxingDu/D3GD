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
    df = align_metrics_for_plotting(history)
    if df.empty:
        print("Warning: No history data to plot.")
        return

    # --- Plot 1: Main Performance Metric ---
    plt.figure(figsize=(10, 6))
    if 'test_mse' in df.columns and df['test_mse'].notna().any():
        valid_mse = df['test_mse'][df['test_mse'] > 0]
        plt.semilogy(valid_mse.index, valid_mse, label='Test MSE', linewidth=2, color='red')
        plt.ylabel('Mean Squared Error (Log Scale)')
        plt.title(f'Convergence: MSE vs Iterations {title_suffix}')
        plot_name = 'mse_vs_iter.png'
    elif 'test_acc' in df.columns and df['test_acc'].notna().any():
        plt.plot(df.index, df['test_acc'], label='Test Accuracy', linewidth=2)
        plt.ylabel('Accuracy (%)')
        plt.title(f'Test Accuracy {title_suffix}')
        plot_name = 'acc_vs_iter.png'
    else:
        return 

    plt.xlabel('Iterations')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, plot_name))
    plt.close()
    
    # --- Plot 2: Consensus Error ---
    if 'consensus_error' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.semilogy(df.index, df['consensus_error'], label='Consensus Error', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Consensus Error (Log Scale)')
        plt.title(f'Consensus Error {title_suffix}')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'consensus_vs_iter.png'))
        plt.close()
    
    # --- Plot 3: Efficiency ---
    if 'comm_volume' in df.columns:
        plt.figure(figsize=(10, 6))
        if 'test_mse' in df.columns and df['test_mse'].notna().any():
            valid_idx = df['test_mse'] > 0
            plt.semilogy(df.loc[valid_idx, 'comm_volume'], df.loc[valid_idx, 'test_mse'], 
                         label='Test MSE', color='purple')
            plt.ylabel('MSE (Log Scale)')
        elif 'test_acc' in df.columns and df['test_acc'].notna().any():
            plt.plot(df['comm_volume'], df['test_acc'], label='Test Accuracy', color='green')
            plt.ylabel('Accuracy (%)')
            
        plt.xlabel('Communication Volume (Floats exchanged)')
        plt.title(f'Efficiency: Metric vs Comm Volume {title_suffix}')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'efficiency_vs_comm.png'))
        plt.close()