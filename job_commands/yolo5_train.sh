#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=train_nuimages_v5_pretrained
#SBATCH --ntasks=1
#SBATCH --mail-user=trunghieu.vuong@sjsu.edu
#SBATCH --mail-type=END
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1


module load/cuda-12.2
echo $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# export GIT_PYTHON_REFRESH=quiet
python3 /home/014105936/yolov5/train.py --img 640 --epochs 20 --data /home/014105936/yolov5/nuimages.yaml --weights /home/014105936/yolov5/yolov5m.pt --name v5_pretrained
