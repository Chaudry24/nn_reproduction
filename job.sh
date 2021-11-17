#!/bin/bash
#SBATCH -J mle_nn_job
#SBATCH -o mle_nn.o%j
#SBATCH -t 96:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

module load python/3.9
python -u some_file_3.py
