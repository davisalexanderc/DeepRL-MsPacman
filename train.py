# train.py

import gymnasium as gym
import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter

# import our custom modules
from common.wrappers import PreprocessAndStackFrames, RewardWrapper
from common.utils import load_config, generate_video
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

    # Create directories if they don't exist
    save_path = config.get('save_path', 'models/dqn_checkpoints/')
    os.makedirs(save_path, exist_ok=True)
    video_path = config.get('video_folder_path', 'videos/')
    os.makedirs(video_path, exist_ok=True)
    
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
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    env.metadata['render_fps'] = 60

    # Apply reward shaping if enabled in config
    if config.get('enable_reward_shaping', False):

        print("--- Reward Shaping Enabled ---")
        
        # Get the reward parameters from config
        enable_time = config.get('enable_time_penalty', False)
        time_penalty = config.get('time_penalty_per_step', 0.0)
        enable_death = config.get('enable_death_penalty', False)
        death_penalty = config.get('death_penalty', 0.0)

        # Print the settings that will be used
        if enable_time:
            print(f"Time Penalty per Step: -{time_penalty}")
        if enable_death:
            print(f"Death Penalty: -{death_penalty}")

        env = RewardWrapper(
            env,
            enable_time_penalty=enable_time,
            time_penalty_per_step=time_penalty,
            enable_death_penalty=enable_death,
            death_penalty=death_penalty
        )
        print("------------------------------")

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

        # Saving, Logging and Visualization
        if (timestep + 1) % 1000 == 0:
            print(f"Timestep: {timestep + 1}/{config['total_timesteps']}")

        if (timestep + 1) % config['save_frequency'] == 0:
            checkpoint_path = f"{config['save_path']}dqn_model_step_{timestep + 1}.pth"
            agent.save(checkpoint_path)

        if config.get('capture_video', False) and (timestep + 1) in config.get('video_capture_steps', []):

            generate_video(
                agent=agent, 
                config=config, 
                timestep=(timestep + 1),
            )

    # --- 5. Clean up ---
    wrapped_env.close()
    print("--- Training Complete. ---")


if __name__ == "__main__":
    main()