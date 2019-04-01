# Function for running cross-validated training
from argparse import Namespace
from logging import Logger
from typing import Tuple
from pprint import pformat
from run_training import run_training
import os
import numpy as np
from parsing import parse_train_args
from create_logger import create_logger


def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """k-fold cross validation"""

    # Set up logger
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir

    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        info(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        os.makedirs(args.save_dir, exist_ok=True)
        model_score = run_training(args, logger)
        all_scores.append(model_score)
    all_scores = np.array(all_scores)

    # Report results
    info(f'{args.num_folds}-fold cross validation')

    # Report scores for each fold
    for fold_num, scores in enumerate(all_scores):
        info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

    # Report scores across models
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)
    info(f'Overall test {args.metric} = {mean_score:.6f} +/- {std_score:.6f}')

    return mean_score, std_score


if __name__ == '__main__':
    args = parse_train_args()
    args.num_tasks = 1
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    cross_validate(args, logger)
