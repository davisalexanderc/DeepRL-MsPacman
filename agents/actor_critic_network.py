# agents/actor_critic_network.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    """
    The neural network for the PPO agent.
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
        super(ActorCriticNetwork, self).__init__()

        # Shared Convolutional Base
        in_channels = input_shape[0]
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
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Critic Head
        self.critic_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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
        - x (torch.Tensor): Input observation tensor.

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
        Get the action distribution for the given input.

        Parameters:
        - x (torch.Tensor): Input observation tensor.

        Returns:
        - Categorical: Action distribution.
        """
        action_logits = self.actor_head(self.conv(x).view(x.size(0), -1))

        action_dist = Categorical(logits=action_logits)
        return action_dist
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the estimated value of the state.

        Parameters:
        - x (torch.Tensor): Input observation tensor.

        Returns:
        - torch.Tensor: Estimated value of the state.
        """
        state_value = self.critic_head(self.conv(x).view(x.size(0), -1))

        return state_value