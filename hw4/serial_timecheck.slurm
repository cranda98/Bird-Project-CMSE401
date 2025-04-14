#!/bin/bash
#SBATCH --job-name=serial_check
#SBATCH --output=serial_timecheck.out
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --ntasks=1

module load gcc openmpi

make revROC
bash timecheck.sh
