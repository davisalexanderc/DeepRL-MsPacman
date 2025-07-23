# test_eval.py

from pathlib import Path
from common.utils import load_config
from evaluate import evaluate_agent

def main():
    print("--- Testing the evaluation pipeline ---")
    
    # --- 1. Define the configuration for the test ---
    
    AGENT_TYPE = "dqn"
    # IMPORTANT: Make sure this checkpoint file actually exists from your last test run!
    # Let's use the 4000-step model.
    MODEL_CHECKPOINT_STEP = 4000
    
    # Define paths
    PROJECT_ROOT = Path.cwd()
    model_path = PROJECT_ROOT / f"models/{AGENT_TYPE}_checkpoints/{AGENT_TYPE}_model_step_{MODEL_CHECKPOINT_STEP}.pth"
    config_path = PROJECT_ROOT / f"configs/{AGENT_TYPE}_config.yaml"
    
    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found at: {model_path}")
        print("Please run a short training session to create a checkpoint first.")
        return
        
    # --- 2. Load the base config and add evaluation-specific settings ---
    config = load_config(config_path)
    
    config['agent'] = AGENT_TYPE
    config['model_path'] = str(model_path)
    config['num_games'] = 2  # Run just 2 games for a quick test
    
    # Crucially, ensure reward shaping is OFF for evaluation
    config['enable_reward_shaping'] = False
    config['level_completion_bonus'] = 0.0
    config['death_penalty'] = 0.0
    
    # --- 3. Call the evaluation function ---
    try:
        raw_results_df, summary_df = evaluate_agent(config)
        
        print("\n--- Test Passed: evaluate_agent ran successfully ---")
        
        print("\nRaw Results per Game:")
        print(raw_results_df)
        
        print("\nSummary Statistics:")
        print(summary_df)
        
    except Exception as e:
        print(f"\n--- Test FAILED with an error: {e} ---")
        # Reraise the exception to get the full traceback for debugging
        raise

if __name__ == "__main__":
    main()