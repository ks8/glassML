#!/bin/bash

source activate pytorch-env
python hyperparameter_optimization.py --train_data_path=train-metadata-endpoints-middle1percent/train-metadata-endpoints-middle1percent.json --val_data_path=val-metadata-endpoints-middle1percent/val-metadata-endpoints-middle1percent.json --test_data_path=test-metadata-endpoints-middle1percent/test-metadata-endpoints-middle1percent.json --batch_size=5 --epochs=10 --dataset_type=classification --attention_pooling --save_dir=endpoints-middle1percent-1headattention-hyperparam-training --num_iters=80 --config_save_path=endpoints-middle1percent-1headattention-optimize/best-hyperparams.json --log_dir=endpoints-middle1percent-1headattention-optimize      



