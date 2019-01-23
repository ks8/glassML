from argparse import Namespace
import logging
from typing import Callable, Iterator

from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from tqdm import tqdm

from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

from dataloader_PyTorch import DataLoader
from GlassBatchMolGraph import GlassBatchMolGraph


def train(model: nn.Module,
          data: DataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: NoamLR,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.
    :param model: Model.
    :param data: A DataLoader.
    :param loss_func: Loss function.
    :param optimizer: Optimizer.
    :param scheduler: A NoamLR learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()

    loss_sum, iter_count = 0, 0
    for batch in tqdm(data, total=len(data)):
        targets = batch.y.float().unsqueeze(1)
        batch = GlassBatchMolGraph(batch)

        # Run model
        model.zero_grad()
        preds = model(batch)
        loss = loss_func(preds, targets)
        loss = loss.sum() / loss.size(0)

        loss_sum += loss.item()
        iter_count += len(batch)

        loss.backward()
        if args.max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lr = scheduler.get_lr()[0]
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            debug("Loss = {:.4e}, PNorm = {:.4f}, GNorm = {:.4f}, lr = {:.4e}".format(loss_avg, pnorm, gnorm, lr))

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                writer.add_scalar('learning_rate', lr, n_iter)

    return n_iter
