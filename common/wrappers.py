"""
Custom Gymnasium wrappers for the Atari environment.

This module contains the unified `AtariWrapper` used for preprocessing
observations and applying custom reward shaping for the Ms. Pac-Man environment.
The wrapper handles frame resizing, grayscale conversion, frame stacking, and
logic for experimental reward modifications like death penalties or level bonuses.
"""

from typing import Tuple, Dict, Any
import cv2
import gymnasium as gym
import numpy as np
from collections import deque

# Set cv2 to not use multithreading, as it can case issues with some RL environments
cv2.setNumThreads(0)

class AtariWrapper(gym.Wrapper):
    """A unified Gymnasium wrapper for Atari environments.

    This wrapper combines several common preprocessing steps and optional reward
    shaping into a single, configurable class. It is designed to be the sole
    wrapper applied to the base Atari environment.

    The wrapper performs the following transformations:
    1.  **Preprocessing:** Converts observations to grayscale and resizes them
        to a specified shape (typically 84x84).
    2.  **Frame Stacking:** Stacks a number of consecutive frames (typically 4)
        along a new channel dimension to provide the agent with temporal
        information (e.g., the direction of ghosts).
    3.  **Reward Shaping (Optional):** Modifies the reward signal based on
        game events like losing a life or completing a level. This is controlled
        by the `enable_reward_shaping` flag in the configuration.
    
    It also injects useful information into the `info` dictionary returned by
    `step()` and `reset()`, such as the `original_reward` and `current_level`.
    """
    def __init__(self, env: gym.Env, config: dict) -> None:
        """Initializes the AtariWrapper.

        Parameters:
        - env (gym.Env): The base Gymnasium environment to wrap.
        - config (Dict[str, Any]): The configuration dictionary, which contains
          parameters for frame dimensions, frame stacking,
          and all reward shaping settings.

        Returns:
            - None
        """
        super().__init__(env)
        
        # Preprocessing Attributes
        self.shape = (config.get('frame_height', 84), config.get('frame_width', 84))
        self.num_stack = config.get('frame_stack', 4)
        self.frames = deque(maxlen=self.num_stack)

        # Observation Space
        obs_shape = (self.num_stack, *self.shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

        # Reward Shaping Attributes
        self.enable_reward_shaping = config.get('enable_reward_shaping', False)
        if self.enable_reward_shaping:
            self.time_penalty = config.get('time_penalty_per_step', 0.0)
            self.death_penalty = config.get('death_penalty', 0.0)
            self.level_bonus = config.get('level_completion_bonus', 0.0)
            
            # Pellet counting state
            self.current_lives = 0
            self.current_level = 0
            self.pellet_count = 0
            self.power_pellet_count = 0
            # Known pellet counts per level in Ms. Pac-Man
            self.pellets_per_level = { 
                0: 220, 1: 220, 2: 240, 3: 240, 4: 240, 5: 238, 6: 238, 
                7: 238, 8: 238, 9: 234, 10: 234, 11: 234, 12: 234 
                                       }
            self.power_pellets_per_level = 4

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Resets the environment and initializes the frame stack.
        This method resets the underlying environment, preprocesses the initial
        observation, and fills the frame stack with the initial frame.

        Parameters:
        - kwargs: Additional arguments to pass to the underlying environment's reset method.

        Returns:
        - obs (np.ndarray): The initial stacked observation after reset.
        - info (dict): Additional information from the environment.
        """

        obs, info = self.env.reset(**kwargs)

        # Reset reward shaping state
        if self.enable_reward_shaping:
            self.current_lives = info.get('lives', 0)
            self.current_level = 0
            self.pellet_count = 0
            self.power_pellet_count = 0
        
        # Clear the frame stack and fill it with the initial frame
        processed_obs = self._preprocess(obs)
        for _ in range(self.num_stack):
            self.frames.append(processed_obs)

        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Takes a step in the environment using the given action.
        This method applies the action to the underlying environment, preprocesses
        the resulting observation, updates the frame stack, and applies optional
        reward shaping.

        Parameters:
        - action (int): The action to take in the environment.

        Returns:
        - obs (np.ndarray): The stacked observation after taking the action.
        - reward (float): The (possibly shaped) reward received after taking the action.
        - terminated (bool): Whether the episode has terminated.
        - truncated (bool): Whether the episode was truncated.
        - info (dict): Additional information from the environment, including
            the original reward and current level.
        """
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Reward shaping logic
        if self.enable_reward_shaping:
            info = info.copy()
            info['original_reward'] = reward
            
            # Apply penalties/bonuses
            reward += self.time_penalty
            if info.get('lives', self.current_lives) < self.current_lives:
                reward += self.death_penalty
            
            # Pellet counting and level bonus
            if info['original_reward'] == 10: self.pellet_count += 1
            elif info['original_reward'] == 50: self.power_pellet_count += 1

            # Check for level completion based on pellet counts
            req_pellets = self.pellets_per_level.get(self.current_level, 234)
            if self.pellet_count >= req_pellets and self.power_pellet_count >= self.power_pellets_per_level:
                reward += self.level_bonus
                self.current_level += 1
                self.pellet_count = 0
                self.power_pellet_count = 0

            # Update state for next step
            self.current_lives = info.get('lives', self.current_lives)
            info['current_level'] = self.current_level
        
        # Preprocessing
        self.frames.append(self._preprocess(obs))

        return self._get_obs(), reward, terminated, truncated, info

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesses a single frame: converts to grayscale and resizes.
        Converts the input RGB frame to grayscale and resizes it to the target
        dimensions specified in the wrapper configuration.

        Parameters:
        - frame (np.ndarray): The raw RGB frame from the environment.

        Returns:
        - frame_resized (np.ndarray): The preprocessed grayscale frame.
        """
        # Convert to grayscale and resize
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_resized = cv2.resize(frame_gray, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)

        return frame_resized

    def _get_obs(self) -> np.ndarray:
        """Returns the current stacked observation.
        Stacks the frames in the frame deque along a new channel dimension to
        create the final observation tensor.

        Parameters:
        - None

        Returns:
        - obs (np.ndarray): The stacked observation.
        """
        
        return np.stack(self.frames, axis=0).astype(np.uint8)