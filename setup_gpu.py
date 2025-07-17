# setup_gpu.py

import torch
import subprocess
import sys

def run_command(command):
    """Helper function to run a command in the shell."""
    try:
        print(f"--- Running Command: {' '.join(command)} ---")
        subprocess.run(command, check=True, shell=True if sys.platform == 'win32' else False)
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
        print("✅ CUDA is already available and configured correctly.")
    else:
        print("❌ CUDA not detected with current PyTorch installation.")
        print("Attempting to install PyTorch with CUDA 12.1 using Conda...")
        
        # Define the conda installation command
        # This is the recommended command from the PyTorch website for a robust installation
        conda_command = [
            "conda", "install", "-y",
            "pytorch", "torchvision", "torchaudio", 
            "pytorch-cuda=12.1", 
            "-c", "pytorch", "-c", "nvidia"
        ]

        if not run_command(conda_command):
            print("\n❌ Failed to install PyTorch with CUDA using Conda.")
            print("Please try running the command manually in your activated conda environment.")
            return # Exit if the installation fails
        
        print("\nInstallation command completed. Verifying...")
        # We can't re-import in the same script easily after a subprocess install.
        # The user should re-run the script or a verification command.
        print("Please re-run this script or start a python interpreter to verify the installation.")
        print("Run `python -c 'import torch; print(torch.cuda.is_available())'` to check.")
        return

    # --- Verification ---
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