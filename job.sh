#!/bin/bash
#SBATCH -J data_generatio_job
#SBATCH -o data_generation.o%j
#SBATCH -t 00:10:00
#SBATCH -N 1 -n 1
##SBATCH -p gpu
##SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/myenv/bin/python -u data_generation_my_idea.py
