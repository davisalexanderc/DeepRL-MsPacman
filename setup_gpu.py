"""
GPU Setup and Verification Script for the PyTorch Environment.

This script checks if the current PyTorch installation can access a CUDA-enabled
GPU. If CUDA is not available, it attempts to install the correct version of
PyTorch with CUDA 12.1 support using Conda.

This utility is designed to be run from the command line within the project's
activated Conda environment to ensure the environment is properly configured for
GPU-accelerated training.

Usage:
    python setup_gpu.py
"""

import torch
import subprocess
import sys

def run_command(command: list[str]) -> bool:
    """Helper function to run a command in the shell.
    
    Parameters:
    - command (list[str]): The command and its arguments as a list of strings.
    
    Returns:
    - bool: True if the command executed successfully, False otherwise.
    """
    print(f"--- Running Command: {' '.join(command)} ---")
    try:
        use_shell = sys.platform == "win32"
        subprocess.run(
            command,
            check=True,
            shell=use_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("--- Command Successful ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- Command Failed: {e} ---")
        return False
    except FileNotFoundError as e:
        print(f"--- Command Failed: {e}. Is conda in your PATH? ---")
        return False

def main():
    """
    Checks for CUDA availability and installs the correct PyTorch build if necessary.
    """
    if torch.cuda.is_available():
        print("CUDA is already available and configured correctly.")
    else:
        print("CUDA not detected with current PyTorch installation.")
        print("Attempting to install PyTorch with CUDA 12.1 using Conda...")
        
        # Construct the conda install command
        conda_command = [
            "conda", "install", "-y",
            "pytorch", "torchvision", "torchaudio", 
            "pytorch-cuda=12.1", 
            "-c", "pytorch", "-c", "nvidia"
        ]

        if not run_command(conda_command):
            print("\nFailed to install PyTorch with CUDA using Conda.")
            print("Please try running the command manually in your activated conda environment.")
            return # Exit if the installation fails
        
        print("\nInstallation command completed. Verifying...")
        # Re-import torch to ensure the new installation is picked up
        print("Please re-run this script or start a python interpreter to verify the installation.")
        print("Run `python -c 'import torch; print(torch.cuda.is_available())'` to check.")
        return

    # Final verification
    print("\n--- Verifying GPU Availability ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version used by PyTorch: {torch.version.cuda}")
    else:
        print("No GPU detected after check.")


if __name__ == "__main__":
    main()