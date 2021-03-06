from argparse import Namespace

import torch
import torch.nn as nn

from model import build_model


def save_checkpoint(model: nn.Module, args: Namespace, path: str):
    """
    Saves a model checkpoint.
    :param model: A PyTorch model.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(path: str, save_dir: str, cuda: bool = False, attention_viz: bool = False) -> nn.Module:
    """
    Loads a model checkpoint and optionally the scaler the model was trained with.
    :param path: Path where checkpoint is saved.
    :param cuda: Whether to move model to cuda.
    :param attention_viz: Whether to visualize attention.
    :param save_dir: Directory to save checkpoints, attention visualizations, or other information
    :return: The loaded model, data scaler, features scaler, and loaded args.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']

    # Update args with current args
    args.cuda = cuda
    args.attention_viz = attention_viz
    args.save_dir = save_dir

    model = build_model(args)
    model.load_state_dict(loaded_state_dict)

    if cuda:
        print('Moving model to cuda')
        model = model.cuda()

    return model
