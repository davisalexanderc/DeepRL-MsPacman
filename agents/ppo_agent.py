# agents/ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Import our custom network
from .actor_critic_network import ActorCriticNetwork

class PPOAgent:
    def __init__(self, config: dict, input_shape: tuple, num_actions: int, device: torch.device):
        """
        Initializes the PPO agent.

        Parameters:
        - config (dict): Configuration dictionary.
        - input_shape (tuple): Shape of the input observation (C, H, W).
        - num_actions (int): Number of possible actions.
        - device (torch.device): Device to run the agent on (CPU or GPU).

        Returns:
        - None
        """

        self.config = config
        self_num_actions = num_actions
        self.device = device

        # Extract Hyperparameters from the config
        self.num_steps = config['num_steps']
        self.num_mini_batches = config['num_mini_batches']
        self.num_epochs = config['num_epochs']
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_epsilon = config['clip_epsilon']
        self.entropy_coeff = config['entropy_coeff']
        self.value_loss_coeff = config['value_loss_coeff']

        # Initialize Network and Optimizer
        self.network = ActorCriticNetwork(input_shape, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Initialize storage for experiences
        self.states = torch.zeros((self.num_steps, *input_shape), dtype=torch.uint8).to(self.device)
        self.actions = torch.zeros((self.num_steps, 1), dtype=torch.long).to(self.device)
        self.log_probs = torch.zeros((self.num_steps, 1)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, 1)).to(self.device)
        self.dones = torch.zeros((self.num_steps, 1)).to(self.device)
        self.values = torch.zeros((self.num_steps, 1)).to(self.device)

        # keep track of the current step
        self.rollout_step_counter = 0

# In the PPOAgent class in agents/ppo_agent.py

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select an action from the policy, and get the action's log probability
        and the state's value from the critic.

        Parameters:
        - state (np.ndarray): The current state of the environment.

        Returns:
        - tuple[int, torch.Tensor, torch.Tensor]: A tuple containing:
            - action (int): The action sampled from the policy.
            - log_prob (torch.Tensor): The log probability of the sampled action.
            - value (torch.Tensor): The value of the state as estimated by the critic.
        """
        # Convert the numpy state to a tensor and normalize
        # Note: We add an unsqueeze(0) to create a batch dimension of 1
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
        
        # We use torch.no_grad() because we are just performing inference, not training
        with torch.no_grad():
            # Get the action distribution from the network's actor head
            action_dist = self.network.get_action_dist(state_tensor)
            
            # Sample an action from the distribution
            action = action_dist.sample()
            
            # Get the log probability of that specific action
            log_prob = action_dist.log_prob(action)
            
            # Get the value of the state from the network's critic head
            value = self.network.get_value(state_tensor)
            
        return action.item(), log_prob, value.squeeze()


    def step(self, state: np.ndarray, action: int, reward: float, done: bool, 
             log_prob: torch.Tensor, value: torch.Tensor) -> None:
        """
        Store a single step of experience into the rollout buffer.

        Parameters:
        - state, action, reward, done: The standard environment transition.
        - log_prob (torch.Tensor): The log probability of the action taken.
        - value (torch.Tensor): The critic's value of the state.
        """
        # Ensure that tensors passed in are on the correct device
        # This is a good defensive programming practice
        action_tensor = torch.tensor([action], device=self.device)
        reward_tensor = torch.tensor([reward], device=self.device)
        done_tensor = torch.tensor([done], device=self.device)

        # Store the experience at the current step index in our pre-allocated tensors
        # The .clone() is important for log_prob and value to avoid issues with the computation graph
        self.states[self.rollout_step_counter] = torch.tensor(state, dtype=torch.uint8, device=self.device)
        self.actions[self.rollout_step_counter] = action_tensor
        self.rewards[self.rollout_step_counter] = reward_tensor
        self.dones[self.rollout_step_counter] = done_tensor
        self.log_probs[self.rollout_step_counter] = log_prob.clone()
        self.values[self.rollout_step_counter] = value.clone()
        
        # Increment the step index, ready for the next experience
        self.rollout_step_counter += 1

    def _calculate_advantages(self, next_state: np.ndarray, next_done: bool) -> torch.Tensor:
        """
        Calculates the advantages for the completed rollout using GAE.

        Parameters:
        - next_state (np.ndarray): The state after the last action taken.
        - next_done (bool): Whether the episode has ended after the last action.
        Returns:
        - torch.Tensor: The calculated advantages for the rollout.
        """

        # We need to get the value of the very last state to "bootstrap" from
        with torch.no_grad():
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            next_value = self.network.get_value(next_state_tensor).squeeze()
        
        # Initialize advantages tensor
        advantages = torch.zeros_like(self.rewards).to(self.device)
        last_gae_lambda = 0
        
        # Loop backwards through the rollout
        for t in reversed(range(self.num_steps)):
            # If the episode ended at this step, the next state's value is 0
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_step_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_step_value = self.values[t + 1]

            # Calculate the TD error (delta)
            delta = self.rewards[t] + self.gamma * next_step_value * next_non_terminal - self.values[t]
            
            # Update the advantage using the GAE formula
            advantages[t] = last_gae_lambda = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            
        return advantages
    
    def learn(self, next_state: np.ndarray, next_done: bool):
        """
        The main learning method. Calculates advantages and updates the network
        for several epochs using mini-batches of the rollout data.
        """
        # --- 1. Calculate Advantages and Returns ---
        advantages = self._calculate_advantages(next_state, next_done)
        # The "return" is the advantage plus the value estimate. It's what the critic tries to predict.
        returns = advantages + self.values

        # --- 2. Flatten the rollout data for easier processing ---
        # We convert the rollout data (num_steps, ...) into a single flat batch
        # Normalizing the states tensor here
        b_states = self.states.float() / 255.0
        b_actions = self.actions.view(-1)
        b_log_probs = self.log_probs.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = self.values.view(-1)

        # --- 3. The PPO Update Loop ---
        # Get the indices for the entire batch
        batch_size = self.num_steps

        mini_batch_size = batch_size // self.num_mini_batches

        batch_indices = np.arange(batch_size)
        
        for epoch in range(self.num_epochs):
            # Shuffle the indices at the start of each epoch
            np.random.shuffle(batch_indices)
            
            # Loop over the data in mini-batches
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = batch_indices[start:end]

                # --- Get the data for the current mini-batch ---
                # Using the shuffled indices, we select a slice from our flattened tensors
                mb_states = b_states[mini_batch_indices]
                mb_actions = b_actions[mini_batch_indices]
                mb_log_probs = b_log_probs[mini_batch_indices]
                mb_advantages = b_advantages[mini_batch_indices]
                mb_returns = b_returns[mini_batch_indices]
                mb_values = b_values[mini_batch_indices]

                # --- Forward pass to get new values from the network ---
                new_dist = self.network.get_action_dist(mb_states)
                new_log_probs = new_dist.log_prob(mb_actions)
                entropy = new_dist.entropy()
                new_values = self.network.get_value(mb_states).view(-1)

                # --- Calculate the Policy Loss (Actor) ---
                log_ratio = new_log_probs - mb_log_probs
                ratio = torch.exp(log_ratio)
                
                # The clipped surrogate objective
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # --- Calculate the Value Loss (Critic) ---
                value_loss = nn.MSELoss()(new_values, mb_returns)
                
                # --- Calculate the Total Loss ---
                loss = (policy_loss - 
                        self.entropy_coeff * entropy.mean() + 
                        self.value_loss_coeff * value_loss)

                # --- Perform Gradient Descent ---
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Clip gradients for stability
                self.optimizer.step()

        # --- 4. Reset the rollout buffer index for the next collection phase ---
        self.rollout_step_counter = 0

    def save(self, path: str) -> None:
        """
        Save the Actor-Critic network's weights to a file.

        Parameters:
        - path (str): The file path to save the model.

        Returns:
        - None
        """
        torch.save(self.network.state_dict(), path)
        print(f"\nModel saved to {path}")

    def load(self, path: str) -> None:
        """
        Load the Actor-Critic network's weights from a file.

        Parameters:
        - path (str): The file path to load the model from.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        print(f"\nModel loaded from {path}")

    from torch.utils.tensorboard import SummaryWriter

    def log_metrics(self, writer: SummaryWriter, global_step: int) -> None:
        """
        Logs agent-specific metrics to TensorBoard.

        Parameters:
        - writer: The TensorBoard writer instance.
        - global_step (int): The current training timestep.
        """
        # We can log the average state value estimated by the critic during the last rollout.
        # This can give a hint about the agent's performance.
        average_value = self.values.mean().item()
        writer.add_scalar("charts/average_value", average_value, global_step=global_step) 