# evaluate.py

import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# Import our custom modules
#from common.wrappers import PreprocessAndStackFrames
from common.utils import setup_environment_and_agent

#from agents import create_agent

def play_one_game(agent: object, env: gym.Env) -> dict:
    """
    Play a single game until the end with the agent acting greedily.
    Tracks and returns desired metrics.
    
    Parameters:
    - agent: The trained agent with a method get_greedy_action(state).
    - env: The environment to play in.
    
    Returns:
    - game_stats (dict): A dictionary containing game statistics such as score, steps, max level reached, and level completion status.
    """
    game_stats = {
        'score': 0,
        'steps': 0,
        'max_level_reached': 0,
        'level_1_completed': False, # Track if level 1 was completed
    }

    state, info = env.reset()
    terminated = False
    truncated = False
    current_level = info.get('current_level', 0)

    while not (terminated or truncated):
        action = agent.get_greedy_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        game_stats['score'] += reward
        game_stats['steps'] += 1

        # Update current level if info provides it
        current_level = info.get('current_level', current_level)
        
        state = next_state

    # Collecting game statistics
    game_stats['max_level_reached'] = current_level
    game_stats['level_1_completed'] = current_level > 0 # Level 1 is index 0

    #print(f"Game finished. Final Score: {game_stats['score']}, Max level: reached {game_stats['max_level_reached']}")

    return game_stats

def calculate_summary_stats(raw_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics from the raw, per-game results DataFrame.

    If the input DataFrame is empty, return an empty DataFrame.

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

    summary_df = pd.DataFrame([summary_dict])
    return summary_df

def evaluate_agent(agent: object, env: gym.Env, num_games: int = 10, show_progress: bool = True) -> pd.DataFrame:
    """
    Evaluate the trained agent on the specified environment and return the results.

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

def find_checkpoints(run_path: Path) -> list[Path]:
    """
    Find all checkpoint files in the given run directory.
    Checkpoints are expected to be named like 'dqn_model_step_1000000.pth'.

    Parameters:
    - run_path: Path object pointing to the run directory.

    Returns:
    - List of Path objects for each checkpoint file, sorted by step number.
    """
    print(f"Searching for checkpoints in: {run_path}")
    if not run_path.exists() or not run_path.is_dir():
        raise FileNotFoundError(f"Run path {run_path} does not exist or is not a directory.")

    # Find all .pth files in the directory
    checkpoint_files = list(run_path.glob("*.pth"))

    # Sort checkpoints based on the timestep extracted from the filename
    checkpoint_files.sort(key=lambda p: int(re.search(r'_(\d+)\.pth$', p.name).group(1)))

    print(f"Found {len(checkpoint_files)} checkpoints in {run_path}")
    return checkpoint_files

def run_batch_evaluation(checkpoint_files: list[Path], config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run batch evaluation on a list of checkpoint files.

    Parameters:
    - checkpoint_files: List of Path objects for each checkpoint file.
    - config_path: Path to the configuration file.

    Returns:
    - Tuple of two DataFrames:
      - final_raw_df: The master DataFrame of all the raw per-game results.
      - final_summary_df: The final summary DataFrame, indexed by timestep.
    """
    wrapped_env, agent, device = setup_environment_and_agent(config)

    all_raw_results = []

    for model_path in checkpoint_files:
        print(f"\nEvaluating Checkpoint: {model_path.name}")

        agent.load(str(model_path))
        agent.q_policy_net.eval()  # Set the policy network to evaluation mode

        raw_df_for_checkpoint = evaluate_single_checkpoint(agent, wrapped_env, num_games=config.get('num_games'))

        # Add metadata columns
        timestep = int(re.search(r'step_(\d+)\.pth$', model_path.name).group(1))
        raw_df_for_checkpoint['timestep'] = timestep
        all_raw_results.append(raw_df_for_checkpoint)

    wrapped_env.close()

    # Concatenate all results into a single DataFrame and calculate summary stats
    final_raw_df = pd.concat(all_raw_results, ignore_index=True)
    final_summary_df = calculate_summary_stats(final_raw_df)

    return final_raw_df, final_summary_df