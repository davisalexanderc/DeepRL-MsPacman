# common/wrappers.py

import cv2
import gymnasium as gym
import numpy as np
from collections import deque

# Set cv2 to not use multithreading, as it can case issues with some RL environments
cv2.setNumThreads(0)

class PreprocessAndStackFrames(gym.Wrapper):
    def __init__(self, env, shape=(84,84), num_stack=4):
        """
        Preprocess and stack frames for the environment.
        
        Parameters:
        - env (gym.Env): The environment to wrap.
        - shape (tuple): The shape to resize frames to.
        - num_stack (int): The number of frames to stack.

        Returns:
        - None
        """

        super(PreprocessAndStackFrames, self).__init__(env)
        self.shape = shape
        self.num_stack = num_stack

        # Initialize a deque to hold the stacked frames
        self.frames = deque(maxlen=num_stack)

        # Reset the environment to get the initial observation
        obs_shape = (num_stack, *shape)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def _preprocess(self, frame):
        """
        Preprocess the frame by converting to grayscale and then resizing it.
        
        Parameters:
        - frame (np.ndarray): The input frame from the environment.
        
        Returns:
        - frame_resized (np.ndarray): The preprocessed frame.
        """

        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize the frame
        frame_resized = cv2.resize(frame_gray, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)

        return frame_resized

    def _get_obs(self):
        """
        Get the current observation by stacking the preprocessed frames.
        
        Parameters:
        - None

        Returns:
        - np.ndarray: The stacked frames as the current observation.
        """
        return np.stack(self.frames, axis=0).astype(np.uint8)
    
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        
        Parameters:
        - **kwargs: Additional keyword arguments for the reset method.
        
        Returns:
        - np.ndarray: The initial observation after resetting the environment.
        """

        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        # Preprocess the initial observation
        processed_obs = self._preprocess(obs)

        for _ in range(self.num_stack):
            self.frames.append(processed_obs)

        return self._get_obs(), info
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Parameters:
        - action: The action to take in the environment.
        
        Returns:
        - obs (np.ndarray): The current observation after taking the action.
        - reward (float): The reward received from the environment.
        - terminated (bool): Whether the episode has terminated.
        - truncated (bool): Whether the episode has been truncated.
        - info (dict): Additional information from the environment.
        """

        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Preprocess the observation and add it to the deque
        self.frames.append(self._preprocess(obs))

        return self._get_obs(), reward, terminated, truncated, info
    
class RewardWrapper(gym.Wrapper):
    """
    A wrapper to modify the reward structure of the environment by 
    adding penalties for time and lives lost.
    """

    def __init__(self, env: gym.Env, 
                 enable_time_penalty: bool = False,
                 time_penalty_per_step: float = 0.0, 
                 enable_death_penalty: bool = False,
                 death_penalty: float = 0.0,
                 enable_level_completion_bonus: bool = False,
                 level_completion_bonus: float = 0.0,
                ) -> None:
        """
        Initialize the RewardWrapper.
        
        Parameters:
        - env (gym.Env): The environment to wrap.
        - enable_time_penalty (bool): Whether to apply a time penalty per step.
        - time_penalty_per_step (float): The penalty to apply for each step taken.
        - enable_death_penalty (bool): Whether to apply a death penalty when a life is lost.
        - death_penalty (float): The penalty to apply when a life is lost.
        - enable_level_completion_bonus (bool): Whether to apply a bonus for level completion.
        - level_completion_bonus (float): The bonus to apply when a level is completed.

        Returns:
        - None
        """

        super().__init__(env)

        # Store parameters
        self.time_penalty_per_step = time_penalty_per_step
        self.death_penalty = death_penalty
        self.enable_time_penalty = enable_time_penalty
        self.enable_death_penalty = enable_death_penalty
        self.enable_level_completion_bonus = enable_level_completion_bonus
        self.level_completion_bonus = level_completion_bonus

        # Initialize current lives
        self.current_lives = 0
        self.current_level = 0

        # Initialize pellet counter for level completion
        self.pellet_eaten_counter = 0
        self.power_pellet_eaten_counter = 0
        self.pellets_per_level = {
            0: 220,  # Level 1
            1: 220,  # Level 2
            2: 240,  # Level 3
            3: 240,  # Level 4
            4: 240,  # Level 5
            5: 238,  # Level 6
            6: 238,  # Level 7
            7: 238,  # Level 8
            8: 238,  # Level 9
            9: 234,  # Level 10
            10: 234, # Level 11
            11: 234, # Level 12
            12: 234, # Level 13
        }
        self.power_pellets_per_level = 4

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take a step in the environment and modify the reward.
        
        Parameters:
        - action: The action to take in the environment.
        
        Returns:
        - obs: The observation after taking the action.
        - modified_reward: The modified reward after applying penalties.
        - terminated: Whether the episode has terminated.
        - truncated: Whether the episode has been truncated.
        - info: Additional information from the environment.
        """

        obs, original_reward, terminated, truncated, info = self.env.step(action)
        # Store original score for logging purposes
        info = info.copy() # Prevents a memory leak by copying the info dict
        info['original_reward'] = original_reward

        # Check if a pellet or power pellet was eaten
        if original_reward == 10:  # Pellet eaten
            self.pellet_eaten_counter += 1
        elif original_reward == 50:  # Power pellet eaten
            self.power_pellet_eaten_counter += 1

        # Check for level completion
        level_cleared = False
        required_pellets = self.pellets_per_level.get(self.current_level, 0)
        required_power_pellets = self.power_pellets_per_level

        if (self.pellet_eaten_counter >= required_pellets and
            self.power_pellet_eaten_counter >= required_power_pellets): # All pellets and power pellets eaten
            level_cleared = True
            self.pellet_eaten_counter = 0
            self.power_pellet_eaten_counter = 0
            self.current_level += 1

            print(f"[RewardWrapper] Level cleared! Current Level: {self.current_level}")

        # Initialize modified reward
        modified_reward = float(original_reward)

        # Apply time penalty if enabled
        if self.enable_time_penalty:
            modified_reward += self.time_penalty_per_step

        # Apply death penalty if enabled and a life was lost
        if self.enable_death_penalty:
            new_lives = info.get('lives', self.current_lives)
            # Compares current lives to new lives, this accounts for life gain and loss
            #if new_lives > self.current_lives:
            #    self.current_lives = new_lives
            if new_lives < self.current_lives:
                modified_reward += self.death_penalty
                #print(f"--- LIFE LOST! Applying {self.death_penalty} penalty. Lives remaining: {new_lives} ---")
            
            self.current_lives = new_lives # Handles life gain and loss

        # Apply level completion bonus if enabled and the level is completed
        if self.enable_level_completion_bonus and level_cleared:
            modified_reward += self.level_completion_bonus
            print(f"[RewardWrapper] Adding Bonus: {self.level_completion_bonus}")

        # Update current level in info
        info['current_level'] = self.current_level

        return obs, modified_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Reset the environment and the current lives counter.
        
        Parameters:
        - **kwargs: Additional keyword arguments for the reset method.
        
        Returns:
        - obs: The initial observation after resetting the environment.
        - info: Additional information from the environment.
        """

        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        info = info.copy() # Prevents a memory leak by copying the info dict

        # Reset current lives
        self.current_lives = info.get('lives', self.current_lives)
        self.pellet_eaten_counter = 0
        self.power_pellet_eaten_counter = 0
        self.current_level = 0

        ###--- Debug ---###
        #print(f"[RewardWrapper] Episode reset. Starting at Level: {self.current_level}")
        info['current_level'] = self.current_level

        return obs, info
    
    