import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob

# ================= CONFIGURATION =================
# OPTION 1: Manual Mode (Overrides Auto-Search if not empty)
# Format: "Legend Label": "Folder Name in results/"
MANUAL_RUNS = {
    # "Di-DGD": "mnist_lenet_heterogeneous_di_dgd_2026...",
}

# OPTION 2: Auto-Search Patterns
# The script looks for folders containing these substrings
AUTO_SEARCH = {
    "Di-DGD (Baseline)": "_di_dgd_",
    "D3GD (Proposed)": "_d3gd_",
    "STL-FW (Topology Learning)": "_stl_fw_",
    "AC-GT (Pruning)": "_ac_gt_"
}

RESULTS_DIR = "results"
# =================================================

def get_run_paths():
    run_paths = {}
    
    # 1. Manual Mode
    if MANUAL_RUNS:
        print(f"--- Using Manual Configuration ---")
        for label, folder in MANUAL_RUNS.items():
            path = os.path.join(RESULTS_DIR, folder)
            if os.path.exists(path):
                run_paths[label] = path
            else:
                print(f"[Error] Folder not found: {path}")
        return run_paths

    # 2. Auto-Search Mode
    print(f"--- Using Auto-Search Mode ---")
    all_runs = glob.glob(os.path.join(RESULTS_DIR, "*"))
    
    for label, substring in AUTO_SEARCH.items():
        candidates = [d for d in all_runs if substring in d and os.path.isdir(d)]
        if not candidates:
            print(f"[Warning] No run found for '{label}' (substring: '{substring}')")
            continue
            
        # Pick the latest run
        candidates.sort(key=os.path.getmtime, reverse=True)
        chosen = candidates[0]
        run_paths[label] = chosen
        print(f"Found latest for '{label}': {os.path.basename(chosen)}")
        
    return run_paths

def load_metrics(run_dir):
    json_path = os.path.join(run_dir, "metrics_raw.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_comparisons():
    runs = get_run_paths()
    if not runs:
        print("No valid runs found. Please run the experiments first.")
        return

    # Define Styles for distinction
    # Format: (Color, Marker, Linestyle)
    styles = {
        "Di-DGD (Baseline)":          ('blue', 'o', '--'),
        "D3GD (Proposed)":            ('red', '*', '-'),    # Solid line for proposed
        "STL-FW (Topology Learning)": ('green', '^', '-.'),
        "AC-GT (Pruning)":            ('purple', 's', ':')
    }
    fallback_colors = ['gray', 'orange', 'cyan', 'brown']

    # Setup Plots
    fig1, ax1 = plt.subplots(figsize=(12, 7)) # Convergence (Epochs)
    fig2, ax2 = plt.subplots(figsize=(12, 7)) # Efficiency (Comm)
    
    for i, (label, run_dir) in enumerate(runs.items()):
        data = load_metrics(run_dir)
        if not data or 'test_acc' not in data:
            continue
            
        # Determine Style
        if label in styles:
            c, m, ls = styles[label]
        else:
            c = fallback_colors[i % len(fallback_colors)]
            m, ls = 'x', '-'
            
        # Extract Data
        acc = np.array(data['test_acc'])
        epochs = np.arange(1, len(acc) + 1)
        
        # --- Plot 1: Accuracy vs Epochs ---
        ax1.plot(epochs, acc, label=label, color=c, marker=m, linestyle=ls, linewidth=2, markersize=5, markevery=5)
        
        # Shade Min/Max if available
        if 'test_acc_min' in data and 'test_acc_max' in data:
            min_len = min(len(acc), len(data['test_acc_min']), len(data['test_acc_max']))
            ax1.fill_between(
                epochs[:min_len], 
                data['test_acc_min'][:min_len], 
                data['test_acc_max'][:min_len], 
                color=c, alpha=0.05
            )

        # --- Plot 2: Accuracy vs Communication ---
        # Prioritize aligned 'test_comm_volume'
        if 'test_comm_volume' in data and len(data['test_comm_volume']) == len(acc):
            comm = np.array(data['test_comm_volume'])
            ax2.plot(comm, acc, label=label, color=c, marker=m, linestyle=ls, linewidth=2, markersize=5)
        elif 'comm_volume' in data:
            comm = np.array(data['comm_volume'])
            # Interpolate comm volume to match epochs if lengths differ
            if len(comm) > len(acc):
                indices = np.linspace(0, len(comm)-1, len(acc)).astype(int)
                comm = comm[indices]
            ax2.plot(comm, acc, label=label, color=c, marker=m, linestyle=ls, linewidth=2)

    # Styling Figure 1
    ax1.set_xlabel("Epochs", fontsize=14)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax1.set_title("Convergence Speed Comparison", fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12, loc='lower right')
    fig1.tight_layout()
    fig1.savefig("comparison_convergence_all.png", dpi=300)
    print("Saved comparison_convergence_all.png")
    
    # Styling Figure 2
    ax2.set_xlabel("Communication Volume (Floats)", fontsize=14)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=14)
    ax2.set_title("Communication Efficiency Comparison", fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12, loc='lower right') 
    fig2.tight_layout()
    fig2.savefig("comparison_efficiency_all.png", dpi=300)
    print("Saved comparison_efficiency_all.png")

if __name__ == "__main__":
    plot_comparisons()