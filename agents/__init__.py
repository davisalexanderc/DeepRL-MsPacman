# agents/__init__.py

import torch
from .dqn_agent import DQNAgent
# from .ppo_agent import PPOAgent  # Uncomment if you have a PPO agent

def create_agent(agent_name: str, config: dict, input_shape: tuple, 
                 num_actions: int, device: torch.device):
    """
    Factory function to create an agent based on the agent name.

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
        # return PPOAgent(config=config, input_shape=input_shape, num_actions=num_actions, device=device)
        raise NotImplementedError("PPO Agent is not implemented yet.")
    else:
        raise ValueError(f"Unknown agent name: {agent_name}. Supported agents: 'dqn', 'ppo'.")