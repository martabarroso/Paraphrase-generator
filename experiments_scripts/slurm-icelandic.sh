#!/bin/bash
#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=icelandic_%j.log

source ~/.bashrc

bash experiments_scripts/icelandic.sh