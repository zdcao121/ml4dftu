#!/bin/bash
#SBATCH --job-name=vasp
#SBATCH --output=vasp.out
#SBATCH -N 3
#SBATCH --ntasks-per-node=40
#SBATCH --mail-type=END
#SBATCH --mail-user=zdcao121@qq.com
#SBATCH --time=72:00:00
#SBATCH -p regular

python ./example.py --which_u 1 0 --br 6 4 --kappa 5 --alpha1 0.25  --alpha2 0.75 --threshold 0.001 --urange 0 10 --elements Mn O --iteration 200 --auto_kpoint 1
