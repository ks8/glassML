# Custom PyTorch method to check for isolated nodes, adapted from PyTorch Geometric repository at
# https://github.com/rusty1s/pytorch_geometric
# Load modules
from __future__ import print_function, division
import torch
from loop_PyTorch import remove_self_loops


def contains_isolated_nodes(edge_index, num_nodes):
	"""Check if there are any isolated nodes"""
	(row, _), _ = remove_self_loops(edge_index)
	return torch.unique(row).size(0) < num_nodes


