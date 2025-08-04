"""
Implements the Experience Replay Buffer for the DQN agent.

This module defines the `ReplayBuffer` class, which stores transitions
(state, action, reward, next_state, done) observed by the agent. Storing and
sampling experiences randomly from this buffer helps to de-correlate the data
used for training, leading to more stable learning.
"""
from typing import Tuple

import random
import numpy as np
from collections import namedtuple, deque

Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """A fixed-size circular buffer for storing and sampling experience tuples.

    This class implements the Experience Replay mechanism, a crucial component for
    stabilizing the training of off-policy reinforcement learning agents like DQN.
    By storing a history of agent-environment interactions (experiences) and
    sampling mini-batches from this history, it addresses two key issues:

    1.  **Breaking Temporal Correlations:** Standard deep learning assumes that
        training data is independent and identically distributed (i.i.d.).
        In RL, consecutive states are highly correlated. By sampling randomly
        from the buffer, we break these correlations, making the training data
        more i.i.d. and stabilizing the learning process.
    2.  **Reusing Past Experiences:** It allows the agent to learn from the same
        experience multiple times, increasing sample efficiency. Rare but important
        experiences are not lost after a single gradient update.

    The buffer is implemented using a `collections.deque` with a fixed `maxlen`,
    which provides efficient O(1) appends and pops from both ends. When the
    buffer reaches its capacity, adding a new experience automatically discards
    the oldest one.
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