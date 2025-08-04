"""
Main training script for training a reinforcement learning agent to play Ms. Pac

This script serves as the main entry point for training different types of agents (e.g., DQN, PPO). It
is driven by command-line arguments and a configuration file, allowing for flexible experimentation.
The training loop is designed to be generic, with agent-specific logic encapsulated within the agent classes.
The script handles environment setup, checkpointing, periodic evaluation, the main training loop and logging 
metrics to TensorBoard.

Usage:
    python train.py --agent <agent_name> --config <path_to_config_yaml>

Examples:
    python train.py --agent dqn --config ./configs/dqn.yaml
    python train.py --agent ppo --config ./configs/ppo.yaml
"""

import torch
import time
import argparse  # Import the argument parsing library
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import gc
from tqdm import trange

# Import our custom modules
from common.utils import load_config, setup_environment_and_agent
from agents import create_agent  # Import our new factory function


def train_agent(config: dict) -> None:
    """
    Initializes and runs the main training loop for the selected agent.

    This function sets up the TensorBoard writer, creates the environment and agent
    using helper functions, and then executes the main training loop for the
    specified number of timesteps. It handles episode management, logging, model
    saving, and periodic garbage collection.

    Parameters:
    - config (dict): Configuration dictionary containing all necessary parameters. 

    Returns:
    - None
    """
    # Initialize TensorBoard writer
    run_name = f"{config['agent']}_{int(time.time())}"
    log_path = config["log_path"] / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard log directory: {log_path}")
    
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create and wrap the environment
    wrapped_env, agent, device = setup_environment_and_agent(config)

    # State initialization
    state, info = wrapped_env.reset()
    episode_reward = 0
    episode_length = 0
    episode_true_score = 0

    # Main training loop
    print("--- Starting Training ---")

    # Progress bar setup
    progress_bar = trange(
        1,
        config['total_timesteps'] + 1,
        ncols=150,  # Force the bar to be 150 characters wide
        unit="step" # Use 'step' instead of the default 'it'
    )

    for timestep in progress_bar:

        # --- Agent Specific Actions and Data Collection ---
        # --- DQN ---
        if config['agent'].lower() == 'dqn':
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)

        # --- PPO ---
        elif config['agent'].lower() == 'ppo':
            action, log_prob, value = agent.act(state)
            next_state, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, done, log_prob, value)


        # Common Logic for State Update and Episode Management
        state = next_state
        
        episode_reward += reward
        episode_length += 1
        episode_true_score += info.get('original_reward', reward) 

        if done:
            # Log episode metrics
            writer.add_scalar("charts/episode_reward", episode_reward, global_step=timestep)
            writer.add_scalar("charts/episode_true_score", episode_true_score, global_step=timestep)
            writer.add_scalar("charts/episode_length", episode_length, global_step=timestep)
            progress_bar.set_description(f"True Score: {episode_true_score}, Ep Length: {episode_length}")
            
            # Reset environment and episode variables
            state, info = wrapped_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_true_score = 0

        # --- Agent-Specific Learning ---
        # --- DQN ---
        if config['agent'].lower() == 'dqn':
            if timestep > config['learning_starts'] and timestep % config['train_frequency'] == 0:
                loss = agent.learn()
                if loss is not None:
                    writer.add_scalar("losses/td_loss", loss, global_step=timestep)
        
            if timestep > config['learning_starts'] and timestep % config['target_update_frequency'] == 0:
                agent.update_target_network()
        
        # --- PPO ---
        elif config['agent'].lower() == 'ppo':
            if agent.rollout_step_counter == config['num_steps']:
                agent.learn(next_state, done)
                
        # Logging and Saving (agent-specific logs are handled inside the agent)
        if timestep % config.get('log_frequency', 1000) == 0:
            agent.log_metrics(writer, timestep)
        
        if timestep % config['save_frequency'] == 0:
            checkpoint_path = config["save_path"] / f"{config['agent']}_model_step_{timestep}.pth"
            agent.save(checkpoint_path)

        # Garbage collection to free up memory
        if timestep % 250 == 0:
            gc.collect()
            if device.type == 'cuda':
                # Clear CUDA cache if using GPU
                torch.cuda.empty_cache()

    # Clean up the environment and writer
    wrapped_env.close()
    writer.close()
    print("--- Training Complete ---")

def parse_args_and_setup_paths(args=None):
    """Parses command-line arguments and sets up save/log paths.

    This function defines the command-line interface for the script, parses the
    provided arguments, and constructs the necessary directory paths for saving

    checkpoints and TensorBoard logs. It ensures the directories exist before
    returning.

    Parameters:
    - args (list of str, optional): A list of command-line arguments to parse.

    Returns:
    - tuple: A tuple containing:
        - parsed_args (argparse.Namespace): The parsed command-line arguments.
        - config (dict): The loaded configuration dictionary.
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train a Reinforcement Learning agent on Ms. Pac-Man.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['dqn', 'ppo'],     # Only allow 'dqn' or 'ppo' for now
        help="The agent to train ('dqn' or 'ppo')."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the configuration YAML file." # e.g., ./configs/dqn.yaml or ./configs/ppo.yaml
    )
    parsed_args = parser.parse_args(args)

    # Load configuration
    config = load_config(parsed_args.config)
    config['agent'] = parsed_args.agent

    # Create a unique run name for this training session
    run_name = f"{parsed_args.agent.upper()}_{int(time.time())}"  # e.g., DQN_1632345678

    # Set up and create logging and model save paths
    project_root = Path.cwd()
    config["log_path"] = project_root / "logs" / run_name
    config["save_path"] = project_root / "models" / run_name

    config["log_path"].mkdir(parents=True, exist_ok=True)
    config["save_path"].mkdir(parents=True, exist_ok=True)

    return parsed_args, config

def main():
    """Main execution function.

    Parses command-line arguments, loads the configuration, sets up paths,
    and starts the training process.

    Parameters:
    - None

    Returns:
    - None
    """
    _, config = parse_args_and_setup_paths()
    train_agent(config)

if __name__ == "__main__":
    main()