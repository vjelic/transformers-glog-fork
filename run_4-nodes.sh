#!/bin/bash

#SBATCH --job-name="hg"
#SBATCH --output="hg_n4_g32_%j.out"
#SBATCH --partition=4CN512C32G4H_4IB_MI250_Ubuntu20
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:29:00

srun -N 4 $@
