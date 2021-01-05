#!/bin/bash
#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=run_exp_%j.log

source ~/.bashrc

bash run-experiments.sh