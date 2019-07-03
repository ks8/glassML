# Main function for running training
# Import modules
from argparse import Namespace
from logging import Logger
import os
from pprint import pformat
import json

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import trange

from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader
from distance_PyTorch import Distance
from evaluate import evaluate
from GlassDataset_PyTorch import GlassDataset
from model import build_model
from nngraph_PyTorch import NNGraph
from augmentation_PyTorch import Augmentation
from train import train
from utils import load_checkpoint, save_checkpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from chemprop.nn_utils import NoamLR, param_count
from parsing import parse_train_args
from chemprop.utils import get_loss_func, get_metric_func

from create_logger import create_logger


def run_training(args: Namespace, logger: Logger = None):
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.
    :param args: args info
    :param logger: logger info
    :return: Optimal average test score (for use in hyperparameter optimization via Hyperopt)
    """

    # Set up logger
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    # Load metadata
    metadata = json.load(open(args.data_path, 'r'))

    # Train/val/test split
    if args.k_fold_split:
        data_splits = []
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        for train_index, test_index in kf.split(metadata):
            splits = [train_index, test_index]
            data_splits.append(splits)
        data_splits = data_splits[args.fold_index]

        if args.use_inner_test:
            train_indices, remaining_indices = train_test_split(data_splits[0], test_size=args.val_test_size,
                                                                random_state=args.seed)
            validation_indices, test_indices = train_test_split(remaining_indices, test_size=0.5,
                                                                random_state=args.seed)

        else:
            train_indices = data_splits[0]
            validation_indices, test_indices = train_test_split(data_splits[1], test_size=0.5, random_state=args.seed)

        train_metadata = list(np.asarray(metadata)[list(train_indices)])
        validation_metadata = list(np.asarray(metadata)[list(validation_indices)])
        test_metadata = list(np.asarray(metadata)[list(test_indices)])

    else:
        train_metadata, remaining_metadata = train_test_split(metadata, test_size=args.val_test_size,
                                                              random_state=args.seed)
        validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=args.seed)

    # Load datasets
    debug('Loading data')
    transform = Compose([Augmentation(args.augmentation_length), NNGraph(args.num_neighbors), Distance(False)])
    train_data = GlassDataset(train_metadata, transform=transform)
    val_data = GlassDataset(validation_metadata, transform=transform)
    test_data = GlassDataset(test_metadata, transform=transform)
    args.atom_fdim = 3
    args.bond_fdim = args.atom_fdim + 1

    # Dataset lengths
    train_data_length, val_data_length, test_data_length = len(train_data), len(val_data), len(test_data)
    debug('train size = {:,} | val size = {:,} | test size = {:,}'.format(
        train_data_length,
        val_data_length,
        test_data_length)
    )

    # Convert to iterators
    train_data = DataLoader(train_data, args.batch_size)
    val_data = DataLoader(val_data, args.batch_size)
    test_data = DataLoader(test_data, args.batch_size)

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(args.metric)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, 'model_{}'.format(model_idx))
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model = load_checkpoint(args.checkpoint_paths[model_idx], args.save_dir)
        else:
            debug('Building model {}'.format(model_idx))
            model = build_model(args)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))

        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(model, args, os.path.join(save_dir, 'model.pt'))

        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.init_lr[model_idx], weight_decay=args.weight_decay[model_idx])

        scheduler = NoamLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=[args.epochs],
            steps_per_epoch=train_data_length // args.batch_size,
            init_lr=args.init_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug('Epoch {}'.format(epoch))

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )

            val_scores = []
            for val_runs in range(args.num_val_runs):

                val_batch_scores = evaluate(
                    model=model,
                    data=val_data,
                    metric_func=metric_func,
                    args=args,
                )

                val_scores.append(np.mean(val_batch_scores))

            # Average validation score
            avg_val_score = np.mean(val_scores)
            debug('Validation {} = {:.3f}'.format(args.metric, avg_val_score))
            writer.add_scalar('validation_{}'.format(args.metric), avg_val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(model, args, os.path.join(save_dir, 'model.pt'))

        # Evaluate on test set using model with best validation score
        info('Model {} best validation {} = {:.3f} on epoch {}'.format(model_idx, args.metric, best_score, best_epoch))
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), args.save_dir, cuda=args.cuda)

        test_scores = []
        for test_runs in range(args.num_test_runs):

            test_batch_scores = evaluate(
                model=model,
                data=test_data,
                metric_func=metric_func,
                args=args
            )

            test_scores.append(np.mean(test_batch_scores))

        # Get accuracy (assuming args.metric is set to AUC)
        metric_func_accuracy = get_metric_func('accuracy')
        test_scores_accuracy = []
        for test_runs in range(args.num_test_runs):

            test_batch_scores = evaluate(
                model=model,
                data=test_data,
                metric_func=metric_func_accuracy,
                args=args
            )

            test_scores_accuracy.append(np.mean(test_batch_scores))

        # Average test score
        avg_test_score = np.mean(test_scores)
        avg_test_accuracy = np.mean(test_scores_accuracy)
        info('Model {} test {} = {:.3f}, test {} = {:.3f}'.format(model_idx, args.metric,
                                                                  avg_test_score, 'accuracy', avg_test_accuracy))
        writer.add_scalar('test_{}'.format(args.metric), avg_test_score, n_iter)

        return avg_test_score, avg_test_accuracy  # For hyperparameter optimization or cross validation use


if __name__ == '__main__':
    args = parse_train_args()
    args.num_tasks = 1
    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    run_training(args, logger)
