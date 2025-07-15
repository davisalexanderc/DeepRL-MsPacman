# train.py


import gymnasium as gym
from common.wrappers import PreprocessAndStackFrames
import numpy as np

def main():
    print("--- Setting up the environment ---")
    # 1. Create the base environment
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")

    # 2. Wrap it with our custom wrapper
    wrapped_env = PreprocessAndStackFrames(env, num_stack=4, shape=(84, 84))
    
    print("Base environment observation space:", env.observation_space.shape)
    print("Wrapped environment observation space:", wrapped_env.observation_space.shape)
    
    # --- Testing the wrapper ---
    print("\n--- Testing the wrapper ---")
    obs, info = wrapped_env.reset()
    
    # Check the shape and data type of the initial observation
    print(f"Shape of initial observation: {obs.shape}")
    print(f"Data type of initial observation: {obs.dtype}")
    
    # Take a random action
    action = wrapped_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    
    # Check the shape and data type after one step
    print(f"\nShape of observation after one step: {next_obs.shape}")
    print(f"Data type of observation after one step: {next_obs.dtype}")
    
    # Verify the values are what we expect (uint8 from 0-255)
    print(f"Min/Max pixel values: {np.min(next_obs)}/{np.max(next_obs)}")

    # Let's see if the frames are actually different after a step
    # We compare the first frame in the stack (oldest) with the last (newest)
    is_same = np.array_equal(next_obs[0], next_obs[-1])
    print(f"Is the oldest frame the same as the newest frame? {'Yes' if is_same else 'No'}")


    wrapped_env.close()
    print("\n--- Test complete ---")

if __name__ == "__main__":
    main()