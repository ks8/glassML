from argparse import Namespace
from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe, Trials
import numpy as np

from parsing_cnn import parse_train_args
from run_training_cnn import run_training

from create_logger import create_logger

import pickle


SPACE = {
    'batch_size': hp.quniform('hidden_size', low=20, high=100, q=10),
    'dropout': hp.quniform('dropout', low=0.0, high=0.5, q=0.05),
}
INT_KEYS = ['batch_size']


def grid_search(args: Namespace):

    # Create logger
    logger = create_logger(name='hyperparameter_optimization', save_dir=os.path.join(args.save_dir, 'optimization_log'),
                           quiet=True)
    train_logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:

        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Update args with hyperparams
        hyper_args = deepcopy(args)
        if args.save_dir is not None:
            folder_name = '_'.join([f'{key}_{value}' for key, value in hyperparams.items()])
            hyper_args.save_dir = os.path.join(hyper_args.save_dir, folder_name)
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)

        logger.info(hyperparams)

        # Train
        test_auc, test_accuracy = run_training(hyper_args, train_logger)

        # Record results
        logger.info(f'{test_auc}' + ' AUC')
        logger.info(f'{test_accuracy}' + ' accuracy')

        results.append({
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'hyperparams': hyperparams,
        })

        return -1 * test_auc

    if args.trials_dir is not None:
        # Load previous trials database
        trials = pickle.load(open(os.path.join(args.trials_dir, 'trials.p'), 'rb'))
    else:
        # Initialize an empty trials database
        trials = Trials()

    # Run TPE algorithm
    fmin(objective, SPACE, algo=tpe.suggest, trials=trials, max_evals=args.max_iters)

    # Save trials
    pickle.dump(trials, open(os.path.join(args.save_dir, 'optimization_log', 'trials.p'), 'wb'))

    # Report best result
    results = [result for result in results if not np.isnan(result['test_auc'])]
    best_result = min(results, key=lambda result: -1 * result['test_auc'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'{best_result["test_auc"]}' + ' AUC')
    logger.info(f'{best_result["test_accuracy"]}' + ' accuracy')

    # Save best hyperparameter settings as JSON config file
    with open(os.path.join(args.save_dir, 'optimization_log', 'best-hyperparams.json'), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    args = parse_train_args()
    grid_search(args)
