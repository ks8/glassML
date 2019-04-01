#!/bin/bash

source activate pytorch-env
python run_training.py --data_path=metadata-PVD-LC/metadata-PVD-LC.json --augmentation_length=0.1 --num_neighbors=2 --batch_size=5 --epochs=2 --dataset_type=classification --hidden_size=870 --depth=3 --dropout=0.5 --save_dir=test-checkpoints --bond_attention --attention_viz



