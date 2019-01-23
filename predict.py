from argparse import Namespace
from typing import List

import torch
import torch.nn as nn

from dataloader_PyTorch import DataLoader
from GlassBatchMolGraph import GlassBatchMolGraph


def predict(model: nn.Module,
            data: DataLoader,
            args: Namespace) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.
    :param model: A model.
    :param data: A DataLoader.
    :param args: Arguments.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    with torch.no_grad():
        model.eval()

        preds = []
        for batch in data:
            # Prepare batch
            batch = GlassBatchMolGraph(batch)

            # Run model
            batch_preds = model(batch)
            batch_preds = batch_preds.data.cpu().numpy()

            preds.extend(batch_preds.tolist())

        return preds