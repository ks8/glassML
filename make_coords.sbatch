#!/bin/bash
#SBATCH --job-name=liquid-middle9percent
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-sandyb
#SBATCH --output=output-liquid-middle9percent
#SBATCH --error=errors-liquid-middle9percent

module unload python
module load Anaconda3/5.1.0
python make_coords.py -data=N2e7 -output=data-endpoints-middle9percent -file_format=coords -timestep=100000 -type=liquid -num_trajectories=10000 -xlo=0.35 -xhi=0.65 -ylo=0.35 -yhi=0.65    



