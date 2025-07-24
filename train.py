# train.py

import gymnasium as gym
import torch
import time
import argparse  # Import the argument parsing library
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import gc

# Import our custom modules
from common.wrappers import PreprocessAndStackFrames, RewardWrapper
from common.utils import load_config, setup_environment_and_agent
from agents import create_agent  # Import our new factory function


def train_agent(config: dict) -> None:
    """
    The main training loop, refactored to be generic.

    Parameters:
    - config (dict): Configuration dictionary containing all necessary parameters. 

    Returns:
    - None
    """
    # --- 1. Set Up ---
    run_name = f"{config['agent']}_{int(time.time())}"
    log_path = config["log_path"] / run_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard log directory: {log_path}")
    
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Create and Wrap the Environment ---
    wrapped_env, agent, device = setup_environment_and_agent(config)

    # --- 4. Main Training Loop ---
    # (This is the same loop you wrote, just adapted slightly to be generic)
    state, info = wrapped_env.reset()
    episode_reward = 0
    episode_length = 0
    episode_true_score = 0

    print("--- Starting Training ---")
    for timestep in range(1, config['total_timesteps'] + 1):

        # # Agent Specific Actions and Data Collection
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
            print(f"Timestep: {timestep}, Ep. Reward: {episode_reward:.2f}, True Score: {episode_true_score}, Ep. Length: {episode_length}")
            writer.add_scalar("charts/episode_reward", episode_reward, global_step=timestep)
            writer.add_scalar("charts/episode_true_score", episode_true_score, global_step=timestep)
            writer.add_scalar("charts/episode_length", episode_length, global_step=timestep)
            
            state, info = wrapped_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_true_score = 0

        # Agent-Specific Learning
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
            #print(f"--- Garbage collection at timestep {timestep} ---")

    # --- 5. Final Cleanup ---
    wrapped_env.close()
    writer.close()
    print("--- Training Complete ---")

def main():
    """
    Loads config from a command-line argument and starts the training process.
    """
    # --- Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Train a Reinforcement Learning agent for Ms. Pac-Man.")
    parser.add_argument("--agent", type=str, required=True, help="The name of the agent to train (e.g., 'dqn').")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config)
    # Add the agent name from the command line to the config for easy access
    config['agent'] = args.agent

    # --- Set up Paths ---
    PROJECT_ROOT = Path.cwd()
    config["save_path"] = PROJECT_ROOT / config.get('save_path', f"models/{args.agent}_checkpoints/")
    config["log_path"] = PROJECT_ROOT / config.get('log_path', 'runs/')
    
    # Create directories if they don't exist
    config["save_path"].mkdir(parents=True, exist_ok=True)
    config["log_path"].mkdir(parents=True, exist_ok=True)
    
    # --- Start Training ---
    train_agent(config)

if __name__ == "__main__":
    main()