#!/bin/bash
#SBATCH --job-name=coords
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-sandyb
#SBATCH --output=output-liquid-test
#SBATCH --error=errors-liquid-test

module unload python
module load Anaconda3/5.1.0
python make_coords.py -data=N2e7 -output=data-test -file_format=coords -timestep=100000 -type=liquid -num_trajectories=100 -xlo=0.45 -xhi=0.55 -ylo=0.45 -yhi=0.55   



