"""
General utility functions shared across the project.

This module contains common helper functions used for tasks like loading
configuration files and setting up the training/evaluation environment and agent.
Keeping these functions here avoids code duplication in `train.py` and
`evaluate.py`.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import gymnasium as gym
import torch
import yaml
import re
from gymnasium.wrappers import RecordVideo

from common.wrappers import AtariWrapper
from agents import create_agent

def load_config(config_path: Path) -> Dict[str, Any]:
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

def setup_environment_and_agent(config: Dict[str, Any]) -> Tuple[gym.Env, object, torch.device]:
    """
    Creates, wraps, and sets up the environment and agent for a run.

    This function handles the complete setup process:
    1. Determines the compute device (CUDA or CPU).
    2. Creates the base Gymnasium environment.
    3. Applies the unified `AtariWrapper` for preprocessing and reward shaping.
    4. Instantiates the appropriate agent (DQN or PPO) using the factory pattern.

    Parameters:
    - config (dict): Configuration dictionary.

    Returns:
    - tuple:
        - wrapped_env (gym.Env): The wrapped Gymnasium environment ready for training/evaluation.
        - agent (object): The instantiated agent (DQN or PPO).
        - device (torch.device): The compute device being used (CUDA or CPU).
    """

    # Determine the compute device (CUDA or CPU)
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create the base environment
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    
    # Wrap the environment with the unified AtariWrapper
    wrapped_env = AtariWrapper(env, config)
    print("Environment created and wrapped with unified AtariWrapper.")

    # Get input shape and number of actions
    input_shape = wrapped_env.observation_space.shape
    num_actions = wrapped_env.action_space.n
    
    # Instantiate the agent
    agent = create_agent(
        agent_name=config['agent'],
        config=config,
        input_shape=input_shape,
        num_actions=num_actions,
        device=device,
    )
    print(f"{config['agent'].upper()} Agent instantiated.")
    
    return wrapped_env, agent, device

def generate_video(agent_type: str, run_name: str, checkpoint_timestep: int, video_filename: Optional[str] = None) -> None:
    """Creates a video recording of a trained agent from a specific checkpoint.

    This function loads a specific agent checkpoint, sets up the appropriate
    environment, and records a video of the agent playing a single episode.

    Parameters:
        - agent_type (str): The type of agent to load ('dqn' or 'ppo').
        - run_name (str): The name of the training run folder (e.g., 'DQN_Run_1').
        - checkpoint_timestep (int): The timestep of the checkpoint to load.
        - video_filename (str, optional): The desired filename for the output video (without extension). If None, a default name is used.
    Returns:
        - None
    """
    print(f"--- Generating video for {run_name} at timestep {checkpoint_timestep} ---")
    project_root = Path.cwd()

    # Load the configuration for the specified agent type
    config_path = project_root / "configs" / f"{agent_type.lower()}_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    config = load_config(config_path)
    config['agent'] = agent_type
    
    # Setup environment and agent
    _, agent, _ = setup_environment_and_agent(config)
    
    # Load the specified checkpoint
    checkpoint_dir = project_root / "models" / f"{agent_type.lower()}_checkpoints" / run_name
    
    # Find the matching checkpoint file. 
    checkpoint_path = None
    for checkpoint_file in checkpoint_dir.glob("*.pth"):
        if f"step_{checkpoint_timestep}.pth" in checkpoint_file.name:
            checkpoint_path = checkpoint_file
            break
    
    if not checkpoint_path:
        raise FileNotFoundError(f"Checkpoint for timestep {checkpoint_timestep} not found in {checkpoint_dir}")
        
    # Load the model weights
    agent.load(checkpoint_path)
    print(f"Successfully loaded checkpoint: {checkpoint_path.name}")

    # Prepare the video directory
    video_dir = project_root / "videos"
    video_dir.mkdir(exist_ok=True)

    # Determine the output filename
    if video_filename is None:
        video_filename = f"{run_name}_step_{checkpoint_timestep}.mp4"
    else:
        video_filename = f"{video_filename}.mp4"
        
    video_path = video_dir / video_filename

    # Create the base environment
    video_env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    
    # Wrap the environment with the unified AtariWrapper
    wrapped_video_env = AtariWrapper(video_env, config)

    # Apply the video recorder wrapper
    final_video_env = RecordVideo(
        env=wrapped_video_env,
        video_folder=str(video_dir),
        name_prefix=video_filename.replace(".mp4", ""),
        episode_trigger=lambda x: x == 0  # Record the very first episode
    )

    # Play one episode and record the video
    agent.set_eval_mode()
    state, _ = final_video_env.reset()
    done = False
    print("Playing episode...")
    while not done:
        action = agent.get_greedy_action(state)
        state, _, terminated, truncated, _ = final_video_env.step(action)
        done = terminated or truncated

    # Cleanup
    final_video_env.close()
    agent.set_train_mode()
    print(f"--- Video saved successfully to: {video_path} ---")

def find_checkpoints(run_path: Path) -> list[Path]:
    """
    Finds and sorts all model checkpoint files in a directory.

    This function scans a given path for files ending in '.pth' and sorts them
    numerically based on the training step number in the filename.

    Checkpoints are expected to be named like 'dqn_model_step_1000000.pth'.

    Parameters:
    - run_path: Path object pointing to the run directory.

    Returns:
    - checkpoint_files (List[Path]): A list of Path objects for each checkpoint file, sorted by step number.
    """

    print(f"Searching for checkpoints in: {run_path}")
    # Ensure the path exists and is a directory
    if not run_path.is_dir():
        print(f"Warning: Checkpoint directory not found at {run_path}")
        return []

    checkpoint_files = list(run_path.glob("*.pth"))
    
    # Define a helper function to extract the step number from the filename
    def get_step_from_path(p: Path) -> int:
        match = re.search(r'step_(\d+)\.pth$', p.name)
        return int(match.group(1)) if match else -1

    # Filter and sort the checkpoint files based on the step number
    valid_checkpoints = [p for p in checkpoint_files if get_step_from_path(p) != -1]
    sorted_checkpoints = sorted(valid_checkpoints, key=get_step_from_path)
    
    # If no valid checkpoints are found, return an empty list
    if not sorted_checkpoints:
        if checkpoint_files:
             print(f"Warning: Found .pth files in {run_path}, but none matched the expected naming convention (e.g., '..._step_123.pth').")
        
    return sorted_checkpoints