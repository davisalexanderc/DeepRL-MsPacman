"""
Defines the deep Q-network (DQN) architecture for the Atari agent.

This module contains the `QNetwork` class, a convolutional neural network (CNN)
that approximates the action-value function (Q-function). It is based on the
architecture described in the original DeepMind paper, "Playing Atari with Deep
Reinforcement Learning."
"""

import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """Convolutional Neural Network for approximating the Q-function.

    This network takes a stack of preprocessed game frames as input and outputs a
    vector of Q-values, one for each possible action in the environment.

    Architecture:
        - Input: (N, C, H, W) tensor, where N is the batch size, C is the number
                 of stacked frames, and H, W are the frame height and width.
        - Three convolutional layers with ReLU activation.
        - A flattening layer.
        - Two fully-connected (linear) layers, with a ReLU activation on the
          first and a linear output on the second.
    """

    def __init__(self, input_shape: tuple, num_actions: int) -> None:
        """
        Initialize the Q-Network.
        
        Parameters:
        - input_shape (tuple): The shape of the input frames (num_stack, height, width).
        - num_actions (int): The number of actions the agent can take.
        
        Returns:
        - None
        """

        super().__init__()
        input_channels = input_shape[0]  # Number of stacked frames

        # Define the convolutional layers
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
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def _get_conv_out_size(self, input_shape: tuple) -> int:
        """
        Calculates the output feature dimension of the convolutional layers.

        This is a helper method that performs a dummy forward pass with a zero tensor
        to dynamically determine the size of the output from the `self.conv` block.
        This makes the network architecture robust to changes in input frame size.
        
        Parameters:
        - input_shape (tuple): The shape of the input frames (num_stack, height, width).
        
        Returns:
        - int: The size of the output from the convolutional layers.
        """
        # Create a dummy input tensor with the same shape as the input frames
        dummy_input = torch.zeros(1, *input_shape)  # Batch size of 1
        conv_out = self.conv(dummy_input)

        return int(torch.prod(torch.tensor(conv_out.size())))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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