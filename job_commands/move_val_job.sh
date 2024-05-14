#!/bin/bash

#SBATCH --job-name=move_val_img
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=trunghieu.vuong@sjsu.edu
#SBATCH --mail-type=END

#Script
python3 /home/014105936/process_val.py 
