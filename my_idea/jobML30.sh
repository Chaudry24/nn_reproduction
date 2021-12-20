#!/bin/bash
#SBATCH -J ml30_my_job
#SBATCH -o ml30_my.o%j
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 1
##SBATCH -p gpu
##SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/myenv/bin/python -u ML30.py
