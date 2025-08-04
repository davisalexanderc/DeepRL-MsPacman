"""
agents Package Initialization

This package contains the implementations of the Reinforcement Learning agents
used in this project.

This __init__.py file exposes the `create_agent` factory function, which provides
a convenient way to instantiate a specific agent based on a string identifier.
This decouples the agent creation logic from the main training script.
"""

from typing import Dict, Any, Tuple

import torch
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent

def create_agent(agent_name: str, config: dict, input_shape: tuple, 
                 num_actions: int, device: torch.device):
    """
    A factory function for creating RL agents.

    This function abstracts the agent instantiation process. Based on the
    `agent_name` string, it returns an initialized instance of the corresponding
    agent class.

    Parameters:
    - agent_name (str): The name of the agent to create.
    - config (dict): Configuration parameters for the agent.
    - input_shape (tuple): The shape of the input frames (num_stack, height, width).
    - num_actions (int): The number of actions the agent can take.
    - device (torch.device): The device to run the agent on (CPU or GPU).
    
    Returns:
    - agent: An instance of the specified agent.
    """

    agent_name = agent_name.lower()

    if agent_name == 'dqn':
        return DQNAgent(
            config=config,
            input_shape=input_shape,
            num_actions=num_actions,
            device=device,
        )
    elif agent_name == 'ppo':
        return PPOAgent(
            config=config,
            input_shape=input_shape,
            num_actions=num_actions,
            device=device,
        )
    else:
        raise ValueError(f"Unknown agent name: {agent_name}. Supported agents: 'dqn', 'ppo'.")