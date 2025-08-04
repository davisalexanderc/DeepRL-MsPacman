"""
Defines the Actor-Critic network architecture for the PPO agent.

This module contains the `ActorCriticNetwork` class, a convolutional neural
network (CNN) that serves both the policy (actor) and the value function (critic).
It uses a shared convolutional base to extract features from the environment's
state, which are then fed into two separate fully-connected "heads."
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    """A CNN with separate heads for the Actor (policy) and Critic (value).

    This network processes image-based states to produce two outputs:
    1.  **Action Logits (Actor):** A vector representing the preference for each
        action. These are used to create a probability distribution over actions.
    2.  **State Value (Critic):** A single scalar value estimating the expected
        return from the current state (V(s)).

    Architecture:
        - A shared convolutional base (identical to the DQN network) extracts a
          feature vector from the input state.
        - The **Actor Head** is a multi-layer perceptron (MLP) that maps the
          feature vector to action logits.
        - The **Critic Head** is an MLP that maps the feature vector to a single
          state-value.
    """
    def __init__(self, input_shape: tuple, num_actions: int) -> None:
        """
        Initializes the Actor-Critic network.

        Parameters:
        - input_shape (tuple): Shape of the input observation (C, H, W).
        - num_actions (int): Number of possible actions.

        Returns:
        - None
        """
        super().__init__()
        in_channels = input_shape[0]

        # Define the convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate the output size after the convolutional layers
        conv_out_size = self._get_conv_out_size(input_shape)

        # Actor Head
        self.actor_head = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1)
        )

    def _get_conv_out_size(self, shape: tuple) -> int:
        """
        Calculate the output size of the convolutional layers.

        Parameters:
        - shape (tuple): Shape of the input observation (C, H, W).

        Returns:
        - int: The output size after the convolutional layers.
        """
        dummy_input = torch.zeros(1, *shape)
        conv_output = self.conv(dummy_input)

        # Calculate the output size
        feature_dim = conv_output.size()[1:]
        return int(torch.prod(torch.tensor(feature_dim)))
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input observation tensor. Shape: (N, C, H, W).

        Returns:
        - tuple: (action_probs, state_value)
            - action_probs (torch.Tensor): Probabilities of each action.
            - state_value (torch.Tensor): Estimated value of the state.
        """
        conv_out = self.conv(x)
        features = conv_out.view(conv_out.size(0), -1)  # Flatten the output

        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)

        return action_logits, state_value
    
    def get_action_dist(self, x: torch.Tensor) -> Categorical:
        """
        A helper method to get the action distribution from the actor head.

        This method performs a forward pass through the shared base and the actor
        head, then wraps the resulting logits in a PyTorch Categorical distribution
        object, which is useful for sampling actions and calculating entropy.

        Parameters:
        - x (torch.Tensor): Input observation tensor.

        Returns:
        - action_dist (Categorical): A categorical distribution over actions.
        """
        features = self.conv(x)
        features_flat = features.view(x.size(0), -1)
        action_logits = self.actor_head(features_flat)

        action_dist = Categorical(logits=action_logits)
        return action_dist
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated value of the state.

        Parameters:
        - x (torch.Tensor): Input observation tensor.

        Returns:
        - state_value (torch.Tensor): Estimated value of the state.
        """
        features = self.conv(x)
        features_flat = features.view(x.size(0), -1)
        state_value = self.critic_head(features_flat)

        return state_value