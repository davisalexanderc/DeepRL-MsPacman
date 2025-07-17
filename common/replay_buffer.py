# comon/replay_buffer.py

import random
import numpy as np
from collections import namedtuple, deque

Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """
    Experience replay buffer to store and sample experiences.
    """

    def __init__(self, capacity, batch_size):
        """
        Initialize the replay buffer.

        Parameters:
        - capacity (int): The maximum size of the buffer.
        - batch_size (int): The batch size for sampling experiences.

        Returns:
        - None
        """

        # Initialize the buffer and parameters
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.

        Parameters:
        - state (np.ndarray): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The next state after taking the action.
        - done (bool): Whether the episode has ended.

        Returns:
        - None
        """

        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self):
        """
        Sample a batch of experiences from the buffer.

        Parameters:
        - None

        Returns:
        - tuple: A tuple containing batches of states, actions, rewards, next_states, and dones.
        """

        experiences_batch = random.sample(self.buffer, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences_batch)

        states_batch = np.array(states)
        actions_batch = np.array(actions)
        rewards_batch = np.array(rewards)
        next_states_batch = np.array(next_states)
        dones_batch = np.array(dones)

        return (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)
    
    def __len__(self):
        """
        Get the current size of the buffer.

        Parameters:
        - None

        Returns:
        - int: The number of experiences currently stored in the buffer.
        """

        return len(self.buffer)