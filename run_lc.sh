#!/bin/bash
#SBATCH --job-name=LC-dataset-N4700-24
#SBATCH --output=output-LC-dataset-N4700-24
#SBATCH --error=errors-LC-dataset-N4700-24
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
python lc_2D-Dan.py --output=LC-dataset-N4700-24 --wall_spring_const=15 --num_turns_deposition 100000 --deposition_runs=120 --substrate_temp 0.18 --num_turns_cooling=4700 --EIS_file=EIS_LC-dataset-N4700-24 --PE_file=PE_LC-dataset-N4700-24 --num_simulations=1 --num_replicas=50
