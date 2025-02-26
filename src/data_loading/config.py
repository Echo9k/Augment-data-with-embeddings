import yaml
import os

class Config:
    def __init__(self, config_file='config.yaml'):
        config_path = os.path.join(os.path.dirname(__file__), '..', config_file)
        with open(config_path, 'r', encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key, default=None):
        # Supports nested keys separated by '.'
        keys = key.split('.')
        val = self.config
        try:
            for k in keys:
                val = val[k]
            return val
        except KeyError:
            return default

# Usage:
config = Config()
max_tokens = config.get("data.max_tokens", 400)