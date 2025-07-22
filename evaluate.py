# evaluate.py

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our custom modules
from common.wrappers import PreprocessAndStackFrames
from common.utils import setup_environment_and_agent
from agents import create_agent

def play_one_game(agent, env):
    """
    Play a single game until the end with the agent acting greedily.
    Tracks and returns desired metrics.
    
    Parameters:
    - agent: The trained agent with a method get_greedy_action(state).
    - env: The environment to play in.
    
    Returns:
    - metrics (dict): A dictionary containing the score, level_completed, and max_level.
    """
    metrics ={
        'score': 0,
        'steps': 0,
        'max_level': 0,
        'level_1_completed': False,
    }

    state, info = env.reset()
    terminated = False
    truncated = False
    current_game_score = 0

    while not (terminated or truncated):
        action = agent.get_greedy_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        current_game_score += reward

        state = next_state
        metrics['steps'] += 1

    # Collecting game statistics
    metrics['score'] = current_game_score
    metrics['max_level'] = env.unwrapped.ale.getRAM()[1]  # Get current level from RAM
    metrics['level_1_completed'] = metrics['max_level'] >= 1

    return metrics

def evaluate_agent(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate the trained agent on the specified environment and return the results.

    Parameters:
    - config (dict): Configuration dictionary containing all necessary parameters.

    Returns:
    - results_df (pd.DataFrame): DataFrame containing the results of each game.
    - summary_df (pd.DataFrame): DataFrame containing summary statistics of the evaluation.

    """
    # Set up the environment and agent
    wrapped_env, agent, device = setup_environment_and_agent(config)

    # Load Trained Model
    agent.load(config["model_path"])
    agent.q_policy_net.eval()  # Set the policy network to evaluation mode
    print(f"Agent loaded from {config['model_path']}")

    all_game_metrics = []
    num_games = config.get('num_evaluation_games', 10)  # Default to 10 games if not specified
    for game_num in range(num_games):
        game_metrics = play_one_game(agent, wrapped_env)
        all_game_metrics.append(game_metrics)

    # Convert the collected stats into a DataFrame
    results_df = pd.DataFrame(all_game_metrics)
    summary_dict = calculate_stats(results_df)
    summary_df = pd.DataFrame([summary_dict])

    print(f"Evaluation completed. Results for {num_games} games:")
    print(summary_df)

    # Cleanup
    wrapped_env.close()

    return results_df, summary_df

def calculate_stats(results_df: pd.DataFrame) -> dict:
    """
    Calculate and return statistics from the evaluation results DataFrame.

    Parameters:
    - results_df (pd.DataFrame): DataFrame containing evaluation results.

    Returns:
    - stats_dict (dict): A dictionary containing calculated statistics.
    """

    level_1_completions = results_df['level_1_completed'].sum()
    total_games = len(results_df)
    level_1_comp_rate = level_1_completions / total_games if total_games > 0 else 0
    mean_level = results_df['max_level'].mean() if 'max_level' in results_df else 0

    stats_dict = {
        'mean_score': results_df['score'].mean(),
        'std_score': results_df['score'].std(),
        'min_score': results_df['score'].min(),
        'Q1_score': results_df['score'].quantile(0.25),
        'Median_score': results_df['score'].median(),
        'Q3_score': results_df['score'].quantile(0.75),
        'max_score': results_df['score'].max(),
        'mean_steps': results_df['steps'].mean(),
        'std_steps': results_df['steps'].std(),
        'level_1_completion_rate': level_1_comp_rate,
        'mean_level_reached': mean_level,
        'max_level_reached': results_df['max_level'].max(),
    }
    
    return stats_dict