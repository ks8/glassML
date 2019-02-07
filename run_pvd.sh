#!/bin/bash
#SBATCH --job-name=pvd-energy
#SBATCH --output=output-pvd-energy
#SBATCH --error=errors-pvd-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
PATH=$PATH:/home/swansonk1/DASH-7-9-2018/md_engine/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/swansonk1/DASH-7-9-2018/md_engine/build
module load cuda/8.0
module load boost/1.62.0+openmpi-1.6+gcc-4.7
python pvd_2D.py --output=pvd-energy --substrate_temp 0.2345 0.214 0.1935 0.173 0.1525 0.132 0.1115 0.091 0.0705 0.05 --num_simulations=10
