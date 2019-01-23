# Custom data objects, adapted from PyTorch Geometric repository at https://github.com/rusty1s/pytorch_geometric
# Load modules
import torch
from loop_PyTorch import contains_self_loops
from isolated_PyTorch import contains_isolated_nodes


# Data class
class Data(object):

	def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None):
		"""
		Custom data class for graph objects. Note: below, 'data' refers to an example instance of Data().
		:param x: (torch.Tensor, preferable type torch.float): node feature matrix, shape [num_nodes, num_node_features]
		:param edge_index: (torch.Tensor of dype torch.long): graph connectivity matrix in COO format, shape [2, num_edges]
		:param edge_attr: (torch.Tensor, preferably type torch.float): edge feature matrix,
		shape [num_edges, num_edge_features]
		:param y: (torch.Tensor): target data, shape arbitrary, but ideally has one dimension only
		:param pos: (torch.Tensor of type torch.float): node position matrix, shape [num_nodes, num_dimensions]
		"""
		self.x = x
		self.edge_index = edge_index
		self.edge_attr = edge_attr
		self.y = y
		self.pos = pos

	@staticmethod
	def from_dict(dictionary):
		"""
		Construct a Data object with custom attributes from a dictionary of keys and items.
		Apply as data = Data.from_dict(<some dictionary>).
		:param dictionary: A dictionary containing custom keys and torch.Tensor items
		:return: The Data() object
		"""
		data = Data()
		for key, item in dictionary.items():
			data[key] = item
		return data

	def __getitem__(self, key):
		"""
		Access object attributes via data['key'] instead of data.key
		:param key: Data() keys such as 'x' or 'pos'
		:return: The corresponding attributes
		"""
		return getattr(self, key)

	def __setitem__(self, key, item):
		"""
		Set object attributes via data['key']=item instead of data.key = item
		:param key: Data() keys such as 'x' or 'pos'
		:param item: Object attribute
		"""
		setattr(self, key, item)

	@property
	def keys(self):
		"""
		data.keys gives a list of object keys (read-only property)
		:return: List of object keys
		"""
		return [key for key in self.__dict__.keys() if self[key] is not None]

	def __len__(self):
		"""
		len(data) gives the number of keys in data
		:return: Number of keys
		"""
		return len(self.keys)

	def __contains__(self, key):
		"""
		'x' in data will return True if x is in data and is not None
		:param key: Data() key such as 'x' or 'pos'
		:return: Boolean
		"""
		return key in self.keys

	def __iter__(self):
		"""
		Allows for iterations such as: for i in data: print(i)
		"""
		for key in sorted(self.keys):
			yield key, self[key]

	def __call__(self, *keys):
		"""
		for i in data(): print(i) will act as __iter__ above; for i in data('x', 'y'): print(i) will only
		iterate over 'x' and 'y' keys
		:param keys: Data() keys such as 'x' or 'pos'
		"""
		for key in sorted(self.keys) if not keys else keys:
			if self[key] is not None:
				yield key, self[key]

	def cat_dim(self, key):
		"""
		Returns the dimension in which the attribute should be concatenated when creating batches.
		:param key: Data() key such as 'x' or 'pos'
		:return: Either -1 for 'edge_index' attributes or 0 otherwise
		"""
		return -1 if key in ['edge_index'] else 0

	@property
	def num_nodes(self):
		"""
		Returns the 0th index of data.x or data.pos for the number of nodes in the system. data.x and data.pos should
		have the same 0th index.
		:return: Number of nodes
		"""
		for key, item in self('x', 'pos'):
			return item.size(0)
		return None

	@property 
	def num_edges(self):
		"""
		Returns the 1th index for data.edge_index and the 0th index for data.edge_attr, corresponding to the number
		of edges in the graph
		:return: Number of edges
		"""
		for key, item in self('edge_index', 'edge_attr'):
			if key == 'edge_index':
				return item.size(1)
			else:
				return item.size(0)
		return None

	@property
	def num_features(self):
		"""
		Number of features, encoded in data.x
		:return: Number of features
		"""
		return 1 if self.x.dim() == 1 else self.x.size(1)

	def contains_isolated_nodes(self):
		"""
		Whether or not the graph contains isolated nodes
		:return: Boolean
		"""
		return contains_isolated_nodes(self.edge_index, self.num_nodes)

	def contains_self_loops(self):
		"""
		Whether or not the graph contains self-loops
		:return: Boolean
		"""
		return contains_self_loops(self.edge_index)

	def apply(self, func, *keys):
		"""
		Apply a function to every key (if *keys is blank) or to a specific set of keys
		:param func: Some function that will act on every element of a data attribute
		:param keys: Data() keys such as 'x' or 'pos'
		:return: The modified Data() object
		"""
		for key, item in self(*keys):
			self[key] = func(item)
		return self

	def contiguous(self, *keys):
		"""
		Apply PyTorch contiguity to every key (if *keys is blank) or to a specific set of keys
		:param keys: Data() keys such as 'x' or 'pos'
		:return: The modified Data() object
		"""
		return self.apply(lambda x: x.contiguous(), *keys)

	def to(self, device, *keys):
		"""
		Move data attributes to device. data.to(device) moves all attributes to device, while
		data.to(device, 'x', 'pos') moves only data.x and data.pos to device
		:param device:
		:param keys:
		:return:
		"""
		return self.apply(lambda x: x.to(device), *keys)

	def __repr__(self):
		"""
		:return: Representation of class in interpreter
		"""
		info = ['{}={}'.format(key, list(item.size())) for key, item in self]
		return '{}({})'.format(self.__class__.__name__, ', '.join(info))




