import yaml
from types import SimpleNamespace

def load_config(path: str):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    if 'lr' in config_dict:
        config_dict['lr'] = float(config_dict['lr'])
    if 'mapping_lr' in config_dict:
        config_dict['mapping_lr'] = float(config_dict['mapping_lr'])
        
    return SimpleNamespace(**config_dict)