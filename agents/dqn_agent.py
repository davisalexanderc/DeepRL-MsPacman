# agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# import components we built
from common.replay_buffer import ReplayBuffer, Experience

class QNetwork(nn.Module):
    """
    The Q-Network for the DQN agent. This is a simple convolutional neural network
    that takes stacked frames as input and outputs Q-values for each action.
    """
    
    def __init__(self, input_shape, num_actions):
        """
        Initialize the Q-Network.
        
        Parameters:
        - input_shape (tuple): The shape of the input frames (num_stack, height, width).
        - num_actions (int): The number of actions the agent can take.
        
        Returns:
        - None
        """

        super(QNetwork, self).__init__()
        input_channels = input_shape[0]  # Number of stacked frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out_size(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def _get_conv_out_size(self, input_shape):
        """
        Calculate the output size of the convolutional layers given the input shape.
        
        Parameters:
        - input_shape (tuple): The shape of the input frames (num_stack, height, width).
        
        Returns:
        - int: The size of the output from the convolutional layers.
        """

        conv_out = self.conv(torch.zeros(1, *input_shape))
        
        return int(torch.prod(torch.tensor(conv_out.size())))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        - x (torch.Tensor): The input tensor containing stacked frames.
        
        Returns:
        - torch.Tensor: The Q-values for each action.
        """
        
        conv_out = self.conv(x)

        # Flatten the output from the convolutional layers
        conv_out = conv_out.view(x.size(0), -1)

        # Pass through the fully connected layers
        q_values = self.fc(conv_out)

        return q_values
    
class DQNAgent:
    """
    The DQN agent that interacts with the environment and learns from experiences.
    """

    def __init__(self, input_shape, num_actions, replay_buffer_capacity, 
                 batch_size, learning_rate, gamma, device):
        """
        Initialize the DQN agent.

        Parameters:
        - input_shape (tuple): The shape of the input frames (num_stack, height, width).
        - num_actions (int): The number of actions the agent can take.
        - replay_buffer_capacity (int): The maximum size of the replay buffer.
        - batch_size (int): The batch size for sampling experiences.
        - learning_rate (float): The learning rate for the optimizer.
        - gamma (float): The discount factor for future rewards.
        - device (torch.device): The device to run the computations on (CPU or GPU).

        Returns:
        - None
        """

        # Initialize parameters
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        # Create the two Q-networks: Policy and Target
        self.q_policy_net = QNetwork(input_shape, num_actions).to(device)
        self.q_target_net = QNetwork(input_shape, num_actions).to(device)

        # Initialize the target network with the same weights as the policy network
        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())
        self.q_target_net.eval()  # Set target network to evaluation mode

        # Create the optimizer
        self.optimizer = optim.Adam(self.q_policy_net.parameters(), lr=learning_rate)

        # Create the replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity, batch_size=batch_size)

    def act(self, state, epsilon):
        """
        Select an action based on the current state and epsilon-greedy policy.

        Parameters:
        - state (torch.Tensor): The current state of the environment.
        - epsilon (float): The probability of selecting a random action.

        Returns:
        - int: The selected action.
        """

        # Decide whether to explore or exploit
        if random.random() < epsilon: # Exploration
            # Select a random action
            action = random.randint(0, self.num_actions - 1)

        else: # Exploitation
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

            with torch.no_grad(): # Disable gradient calculations for action selection
                q_values = self.q_policy_net(state_tensor)

            action = q_values.max(1)[1].item() # Select the action with the highest Q-value

        return action
    
    def learn(self):
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

        return loss.item()
    
    def update_target_network(self):
        """
        Update the target network by copying the weights from the policy network.

        Parameters:
        - None

        Returns:
        - None
        """

        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())