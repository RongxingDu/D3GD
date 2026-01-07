import yaml
import argparse
from typing import Dict, Any
from pathlib import Path

class ConfigMap(dict):
    """
    A dictionary that supports dot notation access.
    Example: config.training.batch_size
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"Config parameter '{name}' not found.")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

def dict_to_map(obj: Any) -> Any:
    """Recursively converts dictionaries to ConfigMap."""
    if isinstance(obj, dict):
        return ConfigMap({k: dict_to_map(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_map(v) for v in obj]
    return obj

class ExperimentConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        # RENAMED: self.data -> self._config to avoid collision with 'data' section
        self._config = self._load_yaml()
        
    def _load_yaml(self) -> ConfigMap:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
                return dict_to_map(config_dict)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

    def get_algorithm_params(self, algo_name: str) -> ConfigMap:
        """
        Retrieves parameters specific to a selected algorithm (e.g., 'd3gd').
        Also merges 'general' algorithm settings like weight_init.
        """
        # Updated references to self._config
        if algo_name not in self._config.algorithm:
            raise ValueError(f"Algorithm '{algo_name}' not defined in config.")
            
        # Base params (like weight_init)
        params = self._config.algorithm.get('general', ConfigMap({})).copy()
        
        # Merge specific params
        specific_params = self._config.algorithm[algo_name]
        params.update(specific_params)
        
        # Add shared init method if not present in specific
        if 'weight_init' in self._config.algorithm:
            params['weight_init'] = self._config.algorithm.weight_init
            
        return params

    def __getattr__(self, name):
        # Delegates attribute access to the internal _config map
        return getattr(self._config, name)

# --- Argument Parser Utility ---

def parse_args():
    parser = argparse.ArgumentParser(description="Decentralized Learning Benchmark")
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='Path to the YAML configuration file (e.g., configs/mnist_lenet.yaml)'
    )
    
    parser.add_argument(
        '--algo', 
        type=str, 
        required=True, 
        choices=['di_dgd', 'd3gd', 'stl_fw', 'ac_gt'],
        help='Algorithm to run'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help='Overwrite random seed'
    )

    return parser.parse_args()