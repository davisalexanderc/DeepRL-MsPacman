# common/utils.py

import yaml

def load_config(config_path):
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - dict: Configuration parameters loaded from the file.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config