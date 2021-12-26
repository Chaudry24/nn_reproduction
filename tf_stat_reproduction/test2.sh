#!/bin/bash
#SBATCH -J test2_job
#SBATCH -o test2.o%j
#SBATCH -t 96:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=240GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/myenv1/bin/python -u test2.py
