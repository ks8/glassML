# Transform a COO format graph to undirected graph, adapted from PyTorch Geometric repository at
# https://github.com/rusty1s/pytorch_geometric
# Load modules
from __future__ import print_function, division
import torch


def to_undirected(edge_index, num_nodes):
	"""Returns an undirected (bidirectional) COO format connectivity matrix from an original matrix given by edge_index"""

	row, col = edge_index
	row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
	unique, inv = torch.unique(row*num_nodes + col, sorted=True, return_inverse=True)
	perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
	perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
	index = torch.stack([row[perm], col[perm]], dim=0)

	return index 
