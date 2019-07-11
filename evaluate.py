from argparse import Namespace
from typing import Callable, List
from tqdm import tqdm

import torch
import torch.nn as nn

from dataloader_PyTorch import DataLoader
from GlassBatchMolGraph import GlassBatchMolGraph

from chemprop.train import evaluate_predictions


def evaluate(model: nn.Module,
             data: DataLoader,
             metric_func: Callable,
             args: Namespace) -> List[float]:
    """
    Evaluates an ensemble of models on a dataset.
    :param model: A model.
    :param data: A GlassDataset.
    :param metric_func: Metric function which takes in a list of targets and a list of predictions.
    :param args: Arguments.
    :return: A list with the score for each task based on `metric_func`.
    """
    targets = []

    with torch.no_grad():
        model.eval()

        preds = []
        for batch in tqdm(data, total=len(data)):

            targets.extend(batch.y.float().unsqueeze(1))

            # Prepare batch
            batch = GlassBatchMolGraph(batch)

            # Run model
            batch_preds = model(batch)
            batch_preds = batch_preds.data.cpu().numpy()

            preds.extend(batch_preds.tolist())

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        metric_func=metric_func,
        dataset_type=args.dataset_type
    )

    return results
