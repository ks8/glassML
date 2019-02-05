#!/bin/bash
#SBATCH --job-name=LC-energies
#SBATCH --time=02:00:00
#SBATCH --partition=depablo-sandyb
#SBATCH --output=output-LC-energies
#SBATCH --error=errors-LC-energies

module unload python
module load Anaconda3/5.1.0
python average_LC_energy.py -data=N2e7 -output=average_LC_EIS -traj_file_start=0 -num_trajectories=10000 -num_steps=200  



