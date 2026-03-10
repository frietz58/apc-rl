# APC-RL
Source code for our ICLR 2026 paper "APC-RL: Exceeding data-driven behavior priors with adaptive policy composition".

# Installation
We provide a dockerfile for your convenience. Assuming you have docker with GPU support set up on your system, you can build the docker image with the following command:
```bash
# Build the docker image
docker build -t apc-rl .
```

Alternatively, install dependencies manually:
```bash
sudo apt update
sudo apt install -y python3-dev build-essential swig

# TODO: Manually install PyTorch with CUDA support matching your system, following https://pytorch.org/get-started/locally/

# Create a virtual environment (optional but recommended)
python3 -m venv apc-venv

# Activate the virtual environment
source apc-venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
```
# Instructions
At a high level, APC pre-trains normalizing flow behavior priors on given offline data, and the uses these priors to together with a prior-free actor to do online learning in a given online environment.

## Datasets
To reproduce our results, download our collected datasets from [here](https://drive.google.com/drive/folders/1hZgpWg8ssxTj8DWRJtZoZT5tZCmhqhVx?usp=sharing) into a `data/` directory in the root of this repository. You can use `gdown` to download the datasets with the following command:
```bash
gdown --folder "https://drive.google.com/drive/folders/1hZgpWg8ssxTj8DWRJtZoZT5tZCmhqhVx?usp=sharing" -O data --remaining-ok --no-check-certificate
```
The datasets should be organized as follows:
```
data/
├── car_racing/
    └── car_racing.pkl
├── maze/
    └── goal_0.pkl
    └── ...
    └── goal_3.pkl
├── kitchen/
    └── microwave.pkl
    └── kettle.pkl
    └── ...
```
Alternatively, use your own data to pre-train the normalizing flow behavior priors.

## Pre-training behavior priors
With offline data in place, the normalizing flow behavior priors can be pre-trained:
```bash
./runner_pretrain.sh
```

## Online learning
Our method is implemented in `online-apc.py`.
The PARROT baseline can be seen like a special instance of APC and can be run by passing the `--parrot` flag.
The other baselines, online SAC, online SAC with IL, and online SAC with IL and Q-filter, can be run via the `online-sac.py` script.

See the runner scripts for concrete examples on how to call these scripts.
```bash
./runner_car.sh
./runner_maze.sh
./runner_kitchen.sh
```

# Citation
```bibtex
@inproceedings{rietz2026apc,
  author       = {Finn Rietz and Pedro Zuidberg Dos Martires and Johannes A. Stork},
  title        = {APC-RL: Exceeding data-driven behavior priors with adaptive policy composition},
  booktitle    = {The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026},
  publisher    = {OpenReview.net},
  year         = {2026},
}
```
