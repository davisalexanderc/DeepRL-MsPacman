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