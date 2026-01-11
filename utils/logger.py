import pandas as pd
import json
import os
import numpy as np
import torch
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config, algo_name):
        self.cfg = config
        self.algo_name = algo_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define directory path
        self.save_dir = os.path.join("results", f"{config.experiment_name}_{algo_name}_{self.timestamp}")
        
        # Ensure it exists immediately
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save Config
        try:
            with open(os.path.join(self.save_dir, "config.json"), "w") as f:
                json.dump(config._config, f, indent=4, default=str)
        except Exception as e:
            print(f"Warning: Failed to save config.json: {e}")
            
        print(f"Logging results to: {self.save_dir}")

    def _align_metrics(self, history_dict):
        """Aligns metrics with different lengths into a single DataFrame."""
        if not history_dict:
            return pd.DataFrame()

        # Identify max length
        lengths = {k: len(v) for k, v in history_dict.items()}
        if not lengths:
            return pd.DataFrame()
        max_len = max(lengths.values())
        
        # Build base dense DataFrame
        data = {}
        for k, v in history_dict.items():
            if len(v) == max_len:
                data[k] = v
        
        df = pd.DataFrame(data)
        
        # Align sparse metrics
        for k, v in history_dict.items():
            if len(v) < max_len and len(v) > 0:
                indices = np.linspace(0, max_len - 1, len(v)).astype(int)
                s = pd.Series(data=v, index=indices)
                df[k] = s.reindex(df.index).ffill()

        return df

    def _convert_to_serializable(self, obj):
        """Helper to convert Torch/Numpy types to standard Python types for JSON."""
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, list):
            return [self._convert_to_serializable(x) for x in obj]
        return obj

    def save_history(self, history_dict):
        """
        Saves history to JSON and CSV. Robust to path issues.
        """
        # 1. Enforce Directory Existence (Fixes FileNotFoundError)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # 2. Sanitize Data (Fixes potential JSON serialization errors)
        clean_history = {k: self._convert_to_serializable(v) for k, v in history_dict.items()}

        # 3. Save JSON
        json_path = os.path.join(self.save_dir, "metrics_raw.json")
        try:
            with open(json_path, "w") as f:
                json.dump(clean_history, f, indent=4)
        except Exception as e:
            print(f"Error saving JSON logs: {e}")

        # 4. Save CSV
        try:
            df = self._align_metrics(clean_history)
            if not df.empty:
                csv_path = os.path.join(self.save_dir, "metrics.csv")
                df.to_csv(csv_path, index_label="iteration")
                return csv_path
        except Exception as e:
            print(f"Error saving CSV logs: {e}")
            
        return json_path
    
    def save_plot(self, plt_figure, filename):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        plt_figure.savefig(os.path.join(self.save_dir, filename))