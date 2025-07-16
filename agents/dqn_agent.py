# agents/dqn_agent.py

import torch
import torch.nn as nn

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