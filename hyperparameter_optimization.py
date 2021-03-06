from argparse import ArgumentParser, Namespace
from copy import deepcopy
import json
from typing import Dict, Union
import os

from hyperopt import fmin, hp, tpe, Trials
import numpy as np

from chemprop.models import build_model
from chemprop.nn_utils import param_count
from parsing import add_train_args, modify_train_args
from run_training import run_training

from create_logger import create_logger

import pickle


SPACE = {
    'hidden_size': hp.quniform('hidden_size', low=300, high=2400, q=100),
    'depth': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.5, q=0.05),
    'ffn_num_layers': hp.quniform('ffn_num_layers', low=1, high=3, q=1),
    'num_neighbors': hp.quniform('num_neighbors', low=1, high=5, q=1),
    'augmentation_length': hp.quniform('augmentation_length', low=0.05, high=0.3, q=0.05)
}
INT_KEYS = ['hidden_size', 'depth', 'ffn_num_layers', 'num_neighbors']


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
        avg_test_score, avg_test_accuracy = run_training(hyper_args, train_logger)

        # Record results
        temp_model = build_model(hyper_args)
        num_params = param_count(temp_model)
        logger.info(f'num params: {num_params:,}')
        logger.info(f'{avg_test_score} {hyper_args.metric}')
        logger.info(f'{avg_test_accuracy}' + ' accuracy')

        results.append({
            'avg_test_score': avg_test_score,
            'avg_test_accuracy': avg_test_accuracy,
            'hyperparams': hyperparams,
            'num_params': num_params
        })

        # Deal with nan
        if np.isnan(avg_test_score):
            if hyper_args.dataset_type == 'classification':
                avg_test_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return (1 if hyper_args.minimize_score else -1) * avg_test_score

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
    results = [result for result in results if not np.isnan(result['avg_test_score'])]
    best_result = min(results, key=lambda result: (1 if args.minimize_score else -1) * result['avg_test_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'num params: {best_result["num_params"]:,}')
    logger.info(f'{best_result["avg_test_score"]} {args.metric}')
    logger.info(f'{best_result["avg_test_accuracy"]}' + ' accuracy')

    # Save best hyperparameter settings as JSON config file
    with open(os.path.join(args.save_dir, 'optimization_log', 'best-hyperparams.json'), 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_train_args(parser)
    parser.add_argument('--max_iters', type=int, default=20,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--trials_dir', type=str, default=None,
                        help='(Optional) Path to a directory where previous trial results are written; hyperopt will'
                             'restart from this info')
    args = parser.parse_args()
    modify_train_args(args)
    args.num_tasks = 1

    grid_search(args)
