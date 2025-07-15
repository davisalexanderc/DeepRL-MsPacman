# debug_env.py

import importlib.metadata
import sys

print(f"--- Running with Python executable: ---\n{sys.executable}\n")

print("--- Checking for 'gymnasium.envs' entry points: ---")
try:
    # This is the command Gymnasium uses to discover external environments.
    entry_points = importlib.metadata.entry_points(group="gymnasium.envs")
    
    if not entry_points:
        print("FAILURE: No entry points found for the 'gymnasium.envs' group.")
        print("This is the root cause of the 'Namespace not found' error.")
    else:
        print("SUCCESS: Found the following entry points:")
        # We are specifically looking for one that mentions 'ale_py'
        found_ale = False
        for ep in entry_points:
            print(f"  - Name: {ep.name}, Value: {ep.value}")
            if 'ale_py' in ep.value:
                found_ale = True
        
        if found_ale:
            print("\nThe ALE entry point was found successfully!")
        else:
            print("\nFAILURE: The 'gymnasium.envs' group was found, but it does not contain the entry point for 'ale_py'.")

except Exception as e:
    print(f"\nAn unexpected error occurred while checking entry points: {e}")