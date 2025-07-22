# common/utils.py

import yaml
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

from common.wrappers import PreprocessAndStackFrames, RewardWrapper
from agents import create_agent

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

def setup_environment_and_agent(config: dict) -> tuple:
    """
    Set up the environment and agent based on the provided configuration.

    Parameters:
    - config (dict): Configuration dictionary containing all necessary parameters.

    Returns:
    - env: The wrapped environment.
    - agent: The instantiated agent.
    - device: The device (CPU or GPU) on which the agent will run.
    """
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    
    if config.get('enable_reward_shaping', False):
        print("Reward Shaping Enabled.")
        env = RewardWrapper(
            env,
            enable_level_completion_bonus=config.get('enable_level_completion_bonus', False),
            level_completion_bonus=config.get('level_completion_bonus', 0.0),
            enable_death_penalty=config.get('enable_death_penalty', False),
            death_penalty=config.get('death_penalty', 0.0)
        )

    wrapped_env = PreprocessAndStackFrames(env, num_stack=4, shape=(84, 84))

    input_shape = wrapped_env.observation_space.shape
    num_actions = wrapped_env.action_space.n
    
    agent = create_agent(
        agent_name=config['agent'],
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
        device=device,
    )
    print(f"{config['agent'].upper()} Agent instantiated.")
    
    return wrapped_env, agent, device