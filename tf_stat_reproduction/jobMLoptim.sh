#!/bin/bash
#SBATCH -J mloptim_tf_job
#SBATCH -o mloptim_tf.o%j
#SBATCH -t 48:00:00
#SBATCH -N 1 -n 1
##SBATCH -p gpu
##SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --mail-user=machaudry@uh.edu
#SBATCH --mail-type=END

/project/jun/machaudr/miniconda3.9/envs/myenv/bin/python -u MLoptim.py
