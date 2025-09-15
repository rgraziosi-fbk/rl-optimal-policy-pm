## Installation

1. Create a new conda environment with Python 3.11: `conda create --name mdp_policy python=3.11`
2. Inside the newly created environment, install required packages: `pip install --no-cache-dir -r requirements.txt`

## Scripts description

- `train.py`: launch this script to run the training pipeline. Make sure to set variables (e.g. DATASET, DEVICE, etc.) to suit your needs
- `predict.py`: launch this script, after a model has been trained, to perform inference
- `dataset.py`: contains the class responsible for creating and managing the offline RL dataset
- `agent.py`: contains the class representing the RL agent
- `model.py`: contains the deep learning neural network used by the agent under the hood

## Training

Run the following command: `python train.py`