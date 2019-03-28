# Custom PyTorch dataset class for loading initial 2D glass data, by Kirk Swanson
# Load modules 
from __future__ import print_function, division
import json
import os
import torch
import numpy as np 
from torch.utils.data import Dataset
from Data_PyTorch import Data 


# Dataset class
class GlassDataset(Dataset):

	def __init__(self, metadata, transform=None):
		"""
		Custom dataset for 2D glass data
		:param metadata: Metadata contents
		:param transform: Transform to apply to the data (can be a Compose() object)
		"""
		super(Dataset, self).__init__()
		self.metadata = metadata
		self.transform = transform

	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, idx):
		""" Output a data object with node features, positions, and target value, transformed as required"""
		coords_file = np.loadtxt(self.metadata[idx]['path'])
		data = Data()
		data.pos = torch.tensor(coords_file[:, 1:], dtype=torch.float)
		data.x = torch.tensor([[x] for x in coords_file[:, 0]], dtype=torch.float)
		if self.metadata[idx]['label'] == 'glass' or self.metadata[idx]['label'] == 'LC':
			data.y = torch.tensor([0])
		else:
			data.y = torch.tensor([1])
		data.uid = torch.tensor([int(self.metadata[idx]['uid'])])

		data = data if self.transform is None else self.transform(data)

		return data

	def __repr__(self):
		return '{}({})'.format(self.__class__.__name__, len(self))



