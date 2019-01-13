#!/bin/bash
#SBATCH --job-name=coords
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-sandyb
#SBATCH --output=output
#SBATCH --error=errors

module unload python
module load Anaconda3/5.1.0
python make_coords.py -data=N2e7 -output=data -file_format=coords -timestep=100000 -type=liquid -num_trajectories=10000   



