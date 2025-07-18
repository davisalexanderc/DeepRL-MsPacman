# common/utils.py

import yaml
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

from common.wrappers import PreprocessAndStackFrames

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

def generate_video(agent, config: dict, timestep: int) -> None:
    """
    Generate a video of the agent playing one episode in a specific environment.

    Parameters:
    - agent: The trained agent with a method get_greedy_action(state).
    - config (dict): Configuration dictionary containing environment and video settings.
    - timestep (int): The current training timestep, used for naming the video folder.

    Returns:
    - None
    """
    video_env = gym.make(config['env_id'], render_mode="rgb_array")
    video_env.metadata['render_fps'] = 60

    video_folder = f"{config['video_folder_path']}{config['env_id'].split('/')[-1]}_step_{timestep}"

    video_env = PreprocessAndStackFrames(video_env, 
                                         num_stack=config.get('frame_stack',4), 
                                         shape=(config.get('frame_height',84), config.get('frame_width',84)))
    
    video_env = RecordVideo(video_env, video_folder=video_folder, episode_trigger=lambda x: x==0)

    print(f"\n--- Generating Video at Timestep {timestep} ---")

    state, info = video_env.reset()
    device = agent.device
    done = False
    while not done:
        action = agent.get_greedy_action(state)
        next_state, reward, terminated, truncated, info = video_env.step(action)
        state = next_state
        done = terminated or truncated

    video_env.close()
    print(f"--- Video saved to {video_folder} ---")
