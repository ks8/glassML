# Compute distances from nearest neighbor graph, adapted from PyTorch Geometric repository at
# https://github.com/rusty1s/pytorch_geometric
# Load modules
from __future__ import print_function, division
import torch 


class Distance(object):

    def __init__(self, cat=False):
        """
        Class for computing NNGraph distances. Initialize as distance = Distance(cat=False), data = distance(data).
        :param cat: (Boolean) Whether or not to concatenate edge distances with existing data.edge_attr
        """
        self.cat = cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist 

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.cat)




