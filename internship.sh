#!/bin/bash

# Script to be run via Slurm for Jérôme CHEN's 2024 Internship

#SBATCH --job-name=Australia_GWL_Model_CHEN
#SBATCH --tasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=3
#SBATCH --mem=24G
#SBATCH --time=10:00:00

source /etc/profile.d/conda.sh
conda init bash
conda activate JeromeEnv
echo "Environment activated"

# make_data.py [-h] [-n {all,no_gwl}] [-mf] [-q] na_threshold {file,month,none}
# make_model.py [-h] [-s] [-q] [-ff FIRST_FILES] [--seed SEED] [--name NAME] {6-1,6-6,7-6} nb_layer layer_size batch_size nb_epoch learning_rate dropout
python3 -u make_data.py 0.2 month -n all -mf
python3 -u make_model.py 6-1 3 256 128 1500 0.0001 0.2 -s --name finetune
