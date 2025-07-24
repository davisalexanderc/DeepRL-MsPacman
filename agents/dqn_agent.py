# agents/dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# import components we built
from common.replay_buffer import ReplayBuffer, Experience
from .q_network import QNetwork
    
class DQNAgent:
    """
    The DQN agent that interacts with the environment and learns from experiences.
    """

    def __init__(self, config: dict, input_shape: tuple, num_actions: int, device: torch.device) -> None:
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
        Select an action based on the current state and epsilon-greedy policy.

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

        loss_values = loss.item()

        # Free up memory to stop memory leaks
        del states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
        del all_q_values, predicted_q_values, next_state_q_values, target_q_values, loss

        return loss_values

    def update_target_network(self) -> None:
        """
        Update the target network by copying the weights from the policy network.

        Parameters:
        - None

        Returns:
        - None
        """

        self.q_target_net.load_state_dict(self.q_policy_net.state_dict())

    def save(self, path: str) -> None:
        """
        Save the policy network's weights to a file.

        Parameters:
        - path (str): The file path to save the model.

        Returns:
        - None
        """
        torch.save(self.q_policy_net.state_dict(), path)
        #print(f"\nModel saved to {path}")

    def get_greedy_action(self, state: np.ndarray) -> int:
        """
        Get the greedy action (action with highest Q-value) for a given state.

        Parameters:
        - state (np.ndarray): The current state of the environment.

        Returns:
        - action (int): The action with the highest Q-value.
        """

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0

        with torch.no_grad():
            q_values = self.q_policy_net(state_tensor)

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
        - writer: The TensorBoard writer instance.
        - global_step (int): The current training timestep.

        Returns:
        - None
        """

        epsilon = np.interp(global_step,
                            [0, self.config['epsilon_decay_duration']],
                            [self.config['epsilon_start'], self.config['epsilon_end']])
        writer.add_scalar("charts/epsilon", epsilon, global_step=global_step)

    def load(self, path: str) -> None:
        """
        Load the policy network's weights from a file.

        Parameters:
        - path (str): The file path to load the model from.

        Returns:
        - None
        """
        state_dict = torch.load(path, map_location=self.device)
        self.q_policy_net.load_state_dict(state_dict)
        self.q_target_net.load_state_dict(state_dict)
        print(f"\nModel loaded from {path}")