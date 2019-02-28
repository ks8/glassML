#!/bin/bash
#SBATCH --job-name=PVD-dataset-7
#SBATCH --output=output-PVD-dataset-7
#SBATCH --error=errors-PVD-dataset-7
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
python pvd_2D-Dan.py --output=PVD-dataset-7 --wall_spring_const=15 --num_turns_deposition 100000 --deposition_runs=120 --substrate_temp 0.18 --EIS_file=EIS_PVD-dataset-7 --PE_file=PE_PVD-dataset-7 --num_simulations=1 --num_replicas=50
