#!/bin/bash
#SBATCH --job-name=hyper-attention
#SBATCH --time=48:00:00
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=output-hyper-attention
#SBATCH --error=errors-hyper-attention

module load singularity
singularity exec --nv /software/src/singularity_images/PyTorch_cuda_9.0.simg bash run_hyperparameter_optimization.sh


