#!/bin/bash

source activate pytorch-env
python run_training.py --train_data_path=train-metadata-endpoints-middle1percent/train-metadata-endpoints-middle1percent.json --val_data_path=val-metadata-endpoints-middle1percent/val-metadata-endpoints-middle1percent.json --test_data_path=test-metadata-endpoints-middle1percent/test-metadata-endpoints-middle1percent.json --num_neighbors=3 --batch_size=5 --epochs=2 --dataset_type=classification --hidden_size=50 --depth=3 --dropout=0.5 --save_dir=test-checkpoints    



