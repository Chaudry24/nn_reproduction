#!/bin/bash
#SBATCH -J nv_my_job
#SBATCH -o nv_my.o%j
#SBATCH -t 96:00:00
#SBATCH -N 1 -n 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/myenv1/bin/python -u NV.py
