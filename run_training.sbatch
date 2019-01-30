#!/bin/bash
#SBATCH --job-name=middle1percent
#SBATCH --time=00:30:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output-middle1percent
#SBATCH --error=errors-middle1percent

module load singularity
singularity exec --nv /software/src/singularity_images/PyTorch_cuda_9.0.simg bash run.sh


