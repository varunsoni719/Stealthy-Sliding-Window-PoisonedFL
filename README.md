# Stealthy-Sliding-Window-PoisonedFL
Code for 'Stealthy Sliding-Window PoisonedFL', demonstrating how to smuggle backdoor attacks past Byzantine-robust aggregation."
# SSW-PoisonedFL & SW-PoisonedFL Analysis

This repository contains the official PyTorch implementation for comparing **SW-PoisonedFL** and **SSW-PoisonedFL** (Backdoor + sliding window) attacks against Federated Learning systems utilizing Multi-Krum defenses. 

The code is strictly deterministic, dynamically scales attack magnitudes, and isolates configurations to allow for complete, bit-for-bit reproducibility.

## Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd <your-repo-folder>
## Install dependencies:
Ensure you have Python 3.9+ installed, then run:

Bash
pip install -r requirements.txt 

## Running the Experiments
By default, the script reads from config.yaml and executes two back-to-back experiments:

Experiment 1: Original SW-PoisonedFL (No Backdoor)

Experiment 2: SSW-PoisonedFL (Combined Backdoor + SW)

To run the pipeline:

Bash
python main.py

Custom Configurations and Output
You can easily point the script to a custom configuration file or specify a different output directory for the resulting weights and plots:

Bash
python main.py --config custom_config.yaml --output_dir ./experiment_results

## Project Structure
main.py - Core federated learning simulation loop, attack vectors, and evaluation.

config.yaml - Hyperparameters for dataset, non-IID splits, and attacker scaling limits.

requirements.txt - Python package dependencies.

outputs/ - Generated plots (ssw_results.png) and model weights (.pt).
