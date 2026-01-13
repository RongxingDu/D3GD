import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
# OPTION 1: Manual Mode (Prioritized)
# Paste the exact folder names from your 'results' directory here.
# Format: "Legend Label": "Folder Name"
MANUAL_RUNS = {
    # Example:
    # "Di-DGD (Baseline)": "mnist_lenet_heterogeneous_di_dgd_20260113_120000",
    # "D3GD (Proposed)": "mnist_lenet_heterogeneous_d3gd_20260113_130000",
}

# OPTION 2: Auto-Detection Mode (Fallback)
# If MANUAL_RUNS is empty, the script will find the latest run containing these substrings.
AUTO_SEARCH = {
    "Di-DGD (Baseline)": "_di_dgd_",
    "D3GD (Proposed)": "_d3gd_"
}

RESULTS_DIR = "results"
# =================================================

def get_run_paths():
    """
    Determines which paths to load based on configuration.
    Returns a dict: {Label: Full_Path}
    """
    run_paths = {}
    
    # 1. Check Manual Configuration first
    if MANUAL_RUNS:
        print(f"--- Using Manual Configuration ({len(MANUAL_RUNS)} runs) ---")
        for label, folder_name in MANUAL_RUNS.items():
            full_path = os.path.join(RESULTS_DIR, folder_name)
            if os.path.exists(full_path):
                run_paths[label] = full_path
            else:
                print(f"[Error] Folder not found: {full_path}")
        return run_paths

    # 2. Fallback to Auto-Search
    print(f"--- Using Auto-Search Mode ---")
    import glob
    all_runs = glob.glob(os.path.join(RESULTS_DIR, "*"))
    
    for label, substring in AUTO_SEARCH.items():
        # Find candidates containing substring
        candidates = [d for d in all_runs if substring in d and os.path.isdir(d)]
        if not candidates:
            print(f"[Warning] No run found for '{label}' (substring: '{substring}')")
            continue
            
        # Sort by time (latest first)
        candidates.sort(key=os.path.getmtime, reverse=True)
        run_paths[label] = candidates[0]
        print(f"Found latest for '{label}': {os.path.basename(candidates[0])}")
        
    return run_paths

def load_metrics(run_dir):
    """Loads metrics_raw.json from the run directory."""
    json_path = os.path.join(run_dir, "metrics_raw.json")
    if not os.path.exists(json_path):
        print(f"[Skip] No metrics_raw.json in {run_dir}")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_comparisons():
    runs = get_run_paths()
    if not runs:
        print("No valid runs found to plot.")
        return

    plt.style.use('default')
    
    # Setup Figure 1: Accuracy vs Epochs
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    # Setup Figure 2: Efficiency (Acc vs Comm)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Color palette
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    color_map = {}
    
    for i, (label, run_dir) in enumerate(runs.items()):
        data = load_metrics(run_dir)
        if not data or 'test_acc' not in data:
            continue
            
        # Assign color
        color = colors[i % len(colors)]
        
        # Extract Data
        acc = np.array(data['test_acc'])
        epochs = np.arange(1, len(acc) + 1)
        
        # --- Plot 1: Accuracy vs Epochs ---
        ax1.plot(epochs, acc, label=label, color=color, linewidth=2, marker='o', markersize=4)
        
        # Add spread (Min/Max) if available
        if 'test_acc_min' in data and 'test_acc_max' in data:
            min_len = min(len(acc), len(data['test_acc_min']), len(data['test_acc_max']))
            ax1.fill_between(
                epochs[:min_len], 
                data['test_acc_min'][:min_len], 
                data['test_acc_max'][:min_len], 
                color=color, alpha=0.1
            )

        # --- Plot 2: Accuracy vs Communication ---
        if 'test_comm_volume' in data and len(data['test_comm_volume']) == len(acc):
            comm = np.array(data['test_comm_volume'])
            ax2.plot(comm, acc, label=label, color=color, linewidth=2, marker='^', markersize=4)
        elif 'comm_volume' in data:
            # Fallback alignment
            comm = np.array(data['comm_volume'])
            if len(comm) > len(acc):
                indices = np.linspace(0, len(comm)-1, len(acc)).astype(int)
                comm = comm[indices]
            ax2.plot(comm, acc, label=label, color=color, linewidth=2, linestyle='--')

    # Finalize Plot 1
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Convergence Comparison", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12)
    fig1.savefig("comparison_convergence.png", dpi=300)
    print("Saved comparison_convergence.png")
    
    # Finalize Plot 2
    ax2.set_xlabel("Communication Volume (Floats)", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.set_title("Communication Efficiency Comparison", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12)
    fig2.savefig("comparison_efficiency.png", dpi=300)
    print("Saved comparison_efficiency.png")

if __name__ == "__main__":
    plot_comparisons()