# DeepRL-MsPacman
An implementation and comparison of DQN and PPO algorithms for the Atari 2600 game Ms. Pac-Man



## Setup and Installation

This project uses Conda for environment management.

1.  **Clone the repository**
    '''bash
    git clone https://github.com/davisalexanderc/DeepRL-MsPacman.git
    cd DeepRL-MsPacman
    '''

2.  **Create and activate the Conda environment
    '''bash
    conda create --name mspacman_rl python=3.11
    conda activate mspacman_rl
    '''

3.  **Install the required packages**
    All dependencies are listed in 'requirements.txt'. Install them using pip
    '''bash
    python -m pip install -r requirements.txt
    '''

4.  **Verify the installation**
    Run the environment test script to ensure everything is setup correctly.
    '''bash
    python train.py
    '''