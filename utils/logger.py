import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config, algo_name):
        self.cfg = config
        self.algo_name = algo_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.save_dir = os.path.join("results", f"{config.experiment_name}_{algo_name}_{self.timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save Config
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            # Helper to dump ConfigMap
            json.dump(config._config, f, indent=4, default=str)
            
        print(f"Logging results to: {self.save_dir}")

    def _align_metrics(self, history_dict):
        """
        Aligns metrics with different lengths into a single DataFrame.
        Assumes sparse metrics (short lists) are roughly evenly spaced.
        """
        if not history_dict:
            return pd.DataFrame()

        # 1. Identify the 'main' timeline (max length)
        lengths = {k: len(v) for k, v in history_dict.items()}
        if not lengths:
            return pd.DataFrame()
            
        max_len = max(lengths.values())
        
        # 2. Build the base DataFrame with dense metrics
        data = {}
        for k, v in history_dict.items():
            if len(v) == max_len:
                data[k] = v
        
        df = pd.DataFrame(data)
        
        # 3. Align and merge sparse metrics (e.g., test_acc)
        for k, v in history_dict.items():
            if len(v) < max_len and len(v) > 0:
                # Infer indices: spread the V points evenly across 0..Max
                indices = np.linspace(0, max_len - 1, len(v)).astype(int)
                
                # Create a Series with these indices
                s = pd.Series(data=v, index=indices)
                
                # Reindex to match the main DF and forward-fill values
                # This creates a "step function" look for accuracy
                s_aligned = s.reindex(df.index).ffill()
                df[k] = s_aligned

        return df

    def save_history(self, history_dict):
        """
        Saves the optimizer's history dictionary to a CSV file.
        Handles unequal list lengths robustly.
        """
        # Save raw JSON first (safest backup)
        json_path = os.path.join(self.save_dir, "metrics_raw.json")
        with open(json_path, "w") as f:
            json.dump(history_dict, f, indent=4)

        # Create aligned DataFrame for CSV
        try:
            df = self._align_metrics(history_dict)
            csv_path = os.path.join(self.save_dir, "metrics.csv")
            df.to_csv(csv_path, index_label="iteration")
            return csv_path
        except Exception as e:
            print(f"Warning: Could not save metrics.csv due to alignment error: {e}")
            return json_path
    
    def save_plot(self, plt_figure, filename):
        plt_figure.savefig(os.path.join(self.save_dir, filename))