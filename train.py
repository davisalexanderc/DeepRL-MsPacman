# train.py

import gymnasium as gym
import torch
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

# import our custom modules
from common.wrappers import PreprocessAndStackFrames
from common.utils import load_config
from agents.dqn_agent import DQNAgent

def main():
    """
    Main function to set up the environment, agent, and start training.
    
    Parameters:
    - None

    Returns:
    - None
    """
    # --- 1. Load Configuration and Set Up ---
    config_path = 'configs/dqn_config.yaml'
    config = load_config(config_path)

    # Set the device for PyTorch
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup TensorBoard ---
    # Create a unique run name based on timestamp and config
    run_name = f"dqn_{config['total_timesteps']}_steps_{int(time.time())}"
    # Create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    # --- 2. Create the Environment ---
    # Create the base environment
    env = gym.make("ALE/MsPacman-v5")
    # Wrap it with our custom preprocessing and frame stacking wrapper
    wrapped_env = PreprocessAndStackFrames(env, num_stack=4, shape=(84, 84))
    
    print("Environment created and wrapped.")

    # --- 3. Instantiate the DQN Agent ---
    # Get environment parameters
    input_shape = wrapped_env.observation_space.shape
    num_actions = wrapped_env.action_space.n

    # Create the agent
    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        replay_buffer_capacity=config['replay_buffer_capacity'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        device=device
    )

    print("DQN Agent instantiated.")
    print("--- Setup Complete. Starting Training... ---")

    # --- 4. The Main Training Loop (to be implemented next) ---
    state, info = wrapped_env.reset()
    episode_reward = 0
    episode_length = 0

    for timestep in range(config['total_timesteps']):
        if timestep < config['epsilon_decay_duration']:
            # Linearly decay epsilon
            epsilon = config['epsilon_start'] - (config['epsilon_start'] - config['epsilon_end']) * (timestep / config['epsilon_decay_duration'])
        else:
            # After decay duration, use final epsilon
            epsilon = config['epsilon_end']

        action = agent.act(state, epsilon)

        # Take the action in the environment
        next_state, reward, terminated, truncated, info = wrapped_env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

        # Store the transition in the replay buffer
        agent.replay_buffer.add(state, action, reward, next_state, done)

        # Update the current state
        state = next_state

        # If episode is done, reset the environment
        if done:
            # Log episode metrics to TensorBoard
            print(f"Timestep: {timestep}, Episode Reward: {episode_reward}, Episode Length: {episode_length}")
            writer.add_scalar("charts/episode_reward", episode_reward, global_step=timestep)
            writer.add_scalar("charts/episode_length", episode_length, global_step=timestep)

            # Reset for the next episode
            episode_reward = 0
            episode_length = 0
            state, info = wrapped_env.reset()

        # Check if we can learn
        if timestep > config['learning_starts'] and timestep % config['train_frequency'] == 0:
            loss = agent.learn()
            # Log the training metrics
            if timestep % 100 == 0:
                writer.add_scalar("losses/td_loss", loss, global_step=timestep)
                writer.add_scalar("charts/epsilon", epsilon, global_step=timestep)

            
        if timestep > config['learning_starts'] and timestep % config['target_update_frequency'] == 0:
            agent.update_target_network()

        # Saving and Logging
        if (timestep + 1) % 1000 == 0:
            print(f"Timestep: {timestep + 1}/{config['total_timesteps']}")

        if (timestep + 1) % config['save_frequency'] == 0:
            agent.save(config['save_path'])

    # --- 5. Clean up ---
    wrapped_env.close()
    print("--- Training Complete. ---")


if __name__ == "__main__":
    main()