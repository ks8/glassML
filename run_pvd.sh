#!/bin/bash
#SBATCH --job-name=2D-energy
#SBATCH --output=output-pvd_2D-energy
#SBATCH --error=errors-pvd_2D-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
PATH=$PATH:/home/swansonk1/DASH-7-9-2018/md_engine/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/swansonk1/DASH-7-9-2018/md_engine/build
module load cuda/8.0
module load boost/1.62.0+openmpi-1.6+gcc-4.7
python pvd_2D.py -output=2D-energy -deposition_runs=55 -substrate_temp=0.2345 -EIS_file=EIS -PE_file=PE 
