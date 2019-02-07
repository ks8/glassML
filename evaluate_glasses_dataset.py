from compose_PyTorch import Compose
from distance_PyTorch import Distance
from nngraph_PyTorch import NNGraph
from GlassDataset_PyTorch import GlassDataset
from utils import load_checkpoint
from dataloader_PyTorch import DataLoader
from evaluate import evaluate
from logging import Logger
from parsing import parse_train_args
from argparse import Namespace
from pprint import pformat
import os
from chemprop.utils import get_metric_func
import numpy as np
from chemprop.nn_utils import param_count


def evaluate_glasses_dataset(args: Namespace, logger: Logger = None):

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    # Load data
    debug('Loading data')
    data = GlassDataset(args.test_data_path, transform=Compose([NNGraph(args.num_neighbors), Distance(False)]))

    # Dataset length
    data_length = len(data)
    debug('data size = {:,}'.format(data_length))

    data = DataLoader(data, args.batch_size)
    metric_func = get_metric_func(args.metric)

    for model_idx in range(args.ensemble_size):

        # Load/build model
        if args.checkpoint_paths is not None:
            debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model = load_checkpoint(args.checkpoint_paths[model_idx])
        else:
            debug('Must provide a checkpoint path!')
            exit(1)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        test_scores = evaluate(
            model=model,
            data=data,
            metric_func=metric_func,
            args=args
        )

        # Average test score
        avg_test_score = np.mean(test_scores)
        info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))


if __name__ == "__main__":
    args = parse_train_args()
    args.num_tasks = 1
    evaluate_glasses_dataset(args)
