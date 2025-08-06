"""
A collection of utility functions for evaluating trained RL agents.

This module provides the core tools needed to run an evaluation pass on a
trained agent. It includes functions to play a single game, evaluate an agent
over multiple games, and calculate summary statistics from the evaluation results.
These functions are designed to be imported and used by a higher-level analysis 
script or notebook.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import our custom modules
from common.utils import setup_environment_and_agent

#from agents import create_agent

def play_one_game(agent: object, env: gym.Env) -> dict:
    """
    Plays a single game episode using the agent's greedy policy.

    This function resets the environment and runs an episode until it terminates
    or is truncated. At each step, it uses the agent's deterministic
    `get_greedy_action` method to select an action. It tracks and returns key
    performance metrics for the episode.
    
    Parameters:
    - agent: The trained agent with a method get_greedy_action(state).
    - env: The environment to play in.
    
    Returns:
    - game_stats (dict): A dictionary containing game statistics such as score, steps, max level reached, and level completion status.
    """
    # Initialize environment and variables
    state, info = env.reset()
    terminated = False
    truncated = False

    game_stats = {
        'score': 0,
        'steps': 0,
        'max_level_reached': info.get('current_level', 0), # Track the highest level reached
        'level_1_completed': False, # Track if level 1 was completed
    }

    # Current level tracking
    while not (terminated or truncated):
        action = agent.get_greedy_action(state)
        state, reward, terminated, truncated, info = env.step(action)

        # Update game statistics
        game_stats['steps'] += 1
        game_stats['score'] += reward

        # Update max level reached
        game_stats['max_level_reached'] = max(info.get('current_level', 0), game_stats['max_level_reached'])

    # Collecting game statistics
    game_stats['level_1_completed'] = game_stats['max_level_reached'] > 0 # Level 1 is index 0

    return game_stats

def calculate_summary_stats(raw_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates a DataFrame of summary statistics from raw game results.

    Given a DataFrame where each row is a game, this function computes aggregate
    metrics like mean, standard deviation, quantiles, and completion rates.

    Parameters:
    - raw_results_df (pd.DataFrame): DataFrame containing the raw results.

    Returns:
    - pd.DataFrame: DataFrame containing the summary statistics.
    """

    summary_cols = [
        'mean_score', 'std_score', 'min_score', 'Q1_score', 'Median_score',
        'Q3_score', 'max_score', 'mean_steps', 'std_steps',
        'level_1_completion_rate', 'mean_level_reached', 'max_level_reached'
    ]
    
    if raw_results_df.empty:
        return pd.DataFrame(columns=summary_cols)
    
    score_stats = raw_results_df['score'].describe()

    summary_dict = {
        'mean_score': score_stats.get('mean', 0),
        'std_score': score_stats.get('std', 0),
        'min_score': score_stats.get('min', 0),
        'Q1_score': score_stats.get('25%', 0),
        'Median_score': score_stats.get('50%', 0),
        'Q3_score': score_stats.get('75%', 0),
        'max_score': score_stats.get('max', 0),
        
        'mean_steps': raw_results_df['steps'].mean(),
        'std_steps': raw_results_df['steps'].std(),
        'level_1_completion_rate': raw_results_df['level_1_completed'].mean() * 100,
        'mean_level_reached': raw_results_df['max_level_reached'].mean(),
        'max_level_reached': raw_results_df['max_level_reached'].max(),
    }

    # Create a DataFrame from the summary statistics
    summary_df = pd.DataFrame([summary_dict])
    return summary_df.reindex(columns=summary_cols)  # Ensure consistent column order

def evaluate_agent(agent: object, env: gym.Env, num_games: int = 10, show_progress: bool = True) -> pd.DataFrame:
    """
    Evaluates a trained agent over a specified number of games.

    This function orchestrates the evaluation by repeatedly calling `play_one_game`.
    It ensures the agent's network is in evaluation mode during the process.

    Parameters:
    - agent: The trained agent with a method get_greedy_action(state).
    - env: The environment to evaluate in.
    - num_games (int): Number of games to evaluate.
    - show_progress (bool): Whether to display a progress bar during evaluation.

    Returns:
    - all_game_stats_df (pd.DataFrame): DataFrame containing the results of each game.
    """

    all_game_stats = []
    agent.set_eval_mode()  # Set the policy network to evaluation mode

    # Use tqdm for progress bar if requested
    game_range = range(num_games)

    if show_progress:
        game_range = tqdm(range(num_games), desc="Evaluating Games", leave=False)

    for _ in game_range:
        game_stats = play_one_game(agent, env)
        all_game_stats.append(game_stats)

    # Convert the collected stats into a DataFrame
    all_game_stats_df = pd.DataFrame(all_game_stats)

    agent.set_train_mode()  # Reset back to training mode

    return all_game_stats_df