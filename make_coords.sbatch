#!/bin/bash
#SBATCH --job-name=images
#SBATCH --time=24:00:00
#SBATCH --partition=depablo-sandyb
#SBATCH --output=output-glass-endpoint-images
#SBATCH --error=errors-glass-endpoint-images

module unload python
module load Anaconda3/5.1.0
python make_coords.py -data=N2e7 -output=data-images-endpoints -file_format=image -timestep=20000000 -type=glass -num_trajectories=5 -trimfrac=0.07 -figsize=250   



