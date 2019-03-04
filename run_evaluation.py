# Main function for running evaluation
# Import modules
from argparse import Namespace
from logging import Logger
from pprint import pformat

import numpy as np

from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader
from distance_PyTorch import Distance
from evaluate import evaluate
from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph
from utils import load_checkpoint

from chemprop.nn_utils import NoamLR, param_count
from parsing import parse_train_args
from chemprop.utils import get_loss_func, get_metric_func

from create_logger import create_logger


def run_evaluation(args: Namespace, logger: Logger = None):
    """
    Evaluates a saved model
    :param args: Set of args
    :param logger: Logger saved in save_dir
    """

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    # Load data
    debug('Loading data')
    test_data = GlassDataset(args.test_data_path, transform=Compose([NNGraph(args.num_neighbors), Distance(False)]))
    args.atom_fdim = 3
    args.bond_fdim = args.atom_fdim + 1

    # Dataset lengths
    test_data_length = len(test_data)
    debug('test size = {:,}'.format(
        test_data_length)
    )

    # Convert to iterators
    test_data = DataLoader(test_data, args.batch_size)

    # Get loss and metric functions
    metric_func = get_metric_func(args.metric)

    # Test ensemble of models
    for model_idx in range(args.ensemble_size):

        # Load/build model
        if args.checkpoint_paths is not None:
            debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model = load_checkpoint(args.checkpoint_paths[model_idx], args.save_dir, cuda=args.cuda,
                                    attention_viz=args.attention_viz)
        else:
            debug('Must specify a model to load')
            exit(1)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))

        # Evaluate on test set using model with best validation score
        test_scores = evaluate(
            model=model,
            data=test_data,
            metric_func=metric_func,
            args=args
        )

        # Average test score
        avg_test_score = np.mean(test_scores)
        info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))


if __name__ == '__main__':
    args = parse_train_args()
    args.num_tasks = 1
    logger = create_logger(name='evaluate', save_dir=args.save_dir, quiet=False)
    run_evaluation(args, logger)
