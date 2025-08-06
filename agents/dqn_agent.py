"""
Implements the Deep Q-Network (DQN) agent.

This module contains the DQNAgent class, which learns to play Ms. Pac-Man
using the DQN algorithm with a replay buffer and a target network for stability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from torch.utils.tensorboard import SummaryWriter

# import components we built
from common.replay_buffer import ReplayBuffer, Experience
from .q_network import QNetwork
    
class DQNAgent:
    """
    A Deep Q-Network agent for playing Atari games.

    This agent implements the DQN algorithm, which uses a deep neural network to
    approximate the optimal action-value function (Q-function). It incorporates
    two key features for stable learning:
    1.  **Experience Replay:** Stores transitions in a ReplayBuffer to break
        the correlation between consecutive experiences.
    2.  **Target Network:** Uses a separate, periodically updated target network
        to provide stable targets for the Bellman equation update, preventing
        oscillations and divergence.

    The agent interacts with the environment using an epsilon-greedy policy for exploration.
    """

    def __init__(self, config: dict, input_shape: tuple, num_actions: int, device: torch.device) -> None:
        """
        Initialize the DQN agent.

        Parameters:
        - config (dict): Configuration dictionary containing hyperparameters.
        - input_shape (tuple): Shape of the input observations (e.g., (4, 84, 84)).
        - num_actions (int): Number of possible actions in the environment.
        - device (torch.device): The compute device (CPU or CUDA) to use.
        
        Returns:
        - None
        """

        # Initialize parameters
        self.config = config
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.device = device
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.learning_rate = config['learning_rate']
        self.timestep = 0  # Track the number of timesteps

        # Create the two Q-networks: Policy and Target
        self.q_policy_net = QNetwork(input_shape, num_actions).to(device)
        self.q_target_net = QNetwork(input_shape, num_actions).to(device)

        # Initialize the target network with the same weights as the policy network
        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())
        self.q_target_net.eval()  # Set target network to evaluation mode

        # Create the optimizer
        self.optimizer = optim.Adam(self.q_policy_net.parameters(), lr=self.learning_rate)

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config['replay_buffer_capacity'], 
                                          batch_size=self.batch_size)

    def act(self, state: torch.Tensor) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        During training, the agent will choose a random action with probability
        epsilon, and the greedy action (the one with the highest Q-value) with
        probability 1-epsilon. Epsilon decays linearly over time.

        Parameters:
        - state (torch.Tensor): The current state of the environment.

        Returns:
        - int: The selected action.
        """

        self.timestep += 1
        epsilon = np.interp(self.timestep,
                            [0, self.config['epsilon_decay_duration']],
                            [self.config['epsilon_start'], self.config['epsilon_end']])

        # Decide whether to explore or exploit
        if random.random() < epsilon: # Exploration
            # Select a random action
            action = random.randint(0, self.num_actions - 1)

        else: # Exploitation
            # Select the action with the highest Q-value
            action = self.get_greedy_action(state)

        return action

    def learn(self) -> float:
        """
        Sample a batch of experiences from the replay buffer and update the policy network.

        Parameters:
        - None

        Returns:
        - float: The loss value after the update.
        """

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device) / 255.0
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device) / 255.0
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute Q-values for current states
        all_q_values = self.q_policy_net(states_tensor)
        predicted_q_values = torch.gather(all_q_values,1, actions_tensor)

        # Compute target Q-values for next states
        with torch.no_grad():
            next_state_q_values = self.q_target_net(next_states_tensor).max(1)[0].unsqueeze(1)

        target_q_values = rewards_tensor + ((1 - dones_tensor) * self.gamma * next_state_q_values)

        # Compute loss
        loss = nn.MSELoss()(predicted_q_values, target_q_values)

        # Perform gradient descent step
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Backpropagate the loss
        self.optimizer.step() # Update the policy network weights

        loss_value = loss.item()

        # Free up memory to stop memory leaks in long training runs. This was observed
        # to be necessary when training for many timesteps.
        del states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
        del all_q_values, predicted_q_values, next_state_q_values, target_q_values, loss

        return loss_value

    def update_target_network(self) -> None:
        """
        Update the target network by copying the weights from the policy network.

        Parameters:
        - None

        Returns:
        - None
        """

        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())

    def save(self, path: Path, timestep: int) -> None:
        """Saves the agent's state (networks, optimizer) to a checkpoint file.

        Parameters:
            - path (Path): The path to the checkpoint file.
            - timestep (int): The current training timestep, saved for resuming.
        
        Returns:
            - None
        """
        checkpoint = {
            'timestep': timestep,
            'network_state_dict': self.q_policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            #'replay_buffer': self.replay_buffer.buffer,
        }
        torch.save(checkpoint, path)

    def get_greedy_action(self, state: np.ndarray) -> int:
        """
        Get the greedy action (action with highest Q-value) for a given state.

        Parameters:
        - state (np.ndarray): The current state of the environment.

        Returns:
        - action (int): The action with the highest Q-value.
        """

        # Convert state to tensor and add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

        with torch.no_grad():
            q_values = self.q_policy_net(state_tensor)

        # Select the action with the highest Q-value
        action = q_values.max(1)[1].item()

        return action
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store the experience in the replay buffer.

        Parameters:
        - state (np.ndarray): The current state of the environment.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (np.ndarray): The next state of the environment.
        - done (bool): Whether the episode has ended.

        Returns:
        - None
        """

        # Store transition in the replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def log_metrics(self, writer: 'SummaryWriter', global_step: int) -> None:
        """
        Log metrics to TensorBoard.

        Parameters:
        - writer (SummaryWriter): The TensorBoard writer instance.
        - global_step (int): The current training timestep.

        Returns:
        - None
        """

        epsilon = np.interp(global_step,
                            [0, self.config['epsilon_decay_duration']],
                            [self.config['epsilon_start'], self.config['epsilon_end']])
        
        writer.add_scalar("charts/epsilon", epsilon, global_step=global_step)

    def load(self, path: Path) -> int:
        """Loads an agent's state from a checkpoint for resuming.

        This method is backwards-compatible. It can load old checkpoints (weights
        only) by inferring the timestep from the filename, and new checkpoints
        (dict with optimizer state) by reading the timestep from the file.

        Parameters:
            - path (Path): The path to the checkpoint file.
        
        Returns:
            - start_timestep (int): The timestep from which to resume training.
        """
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            # New Checkpoint Format
            self.q_policy_net.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_timestep = checkpoint.get('timestep', 0)

            print(f"Loaded new-style checkpoint. Resuming from timestep {start_timestep}.")
        else:
            # Old Checkpoint Format (backwards compatibility, weights only)
            self.q_policy_net.load_state_dict(checkpoint)
            match = re.search(r'step_(\d+)\.pth$', path.name)
            
            if match:
                start_timestep = int(match.group(1))
                print(f"Loaded old-style checkpoint. Inferred timestep {start_timestep}.")
                print("NOTE: Optimizer state not loaded. Will start with a fresh optimizer.")
            else:
                start_timestep = 0
                print("Loaded old-style checkpoint. Could not infer timestep. Starting from 0.")

        # Always sync the target network after loading new weights to the policy network
        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())

        return start_timestep

    def set_eval_mode(self) -> None:
        """
        Set the policy network to evaluation mode.

        Parameters:
        - None

        Returns:
        - None
        """
        self.q_policy_net.eval()

    def set_train_mode(self) -> None:
        """
        Set the policy network to training mode.

        Parameters:
        - None

        Returns:
        - None
        """
        self.q_policy_net.train()