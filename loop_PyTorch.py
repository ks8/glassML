# Custom PyTorch loop methods, adapted from PyTorch Geometric repository at
# https://github.com/rusty1s/pytorch_geometric
# Load modules
from __future__ import print_function, division
import torch


def contains_self_loops(edge_index):
	"""Returns a boolean for existence of self-loops in the graph"""
	row, col = edge_index
	mask = row == col
	return mask.sum().item() > 0


def remove_self_loops(edge_index, edge_attr=None):
	"""Remove self-loops from the edge_index and edge_attr attributes"""
	row, col = edge_index
	mask = row != col
	edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
	mask = mask.expand_as(edge_index)
	edge_index = edge_index[mask].view(2, -1)

	return edge_index, edge_attr


