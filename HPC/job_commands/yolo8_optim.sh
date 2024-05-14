#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=train_nuimages_yolo8_optim
#SBATCH --ntasks=1
#SBATCH --mail-user=trunghieu.vuong@sjsu.edu
#SBATCH --mail-type=END
#SBATCH --time=24:10:00
#SBATCH --gres=gpu


module load/cuda-12.2

export GIT_PYTHON_REFRESH=quiet
python3 /home/014105936/scripts/yolo8_pretrain_optim.py 
