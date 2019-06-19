from scipy.misc import imread, imresize, imrotate
import numpy as np

class Generator():
	def __init__(self, metadata, im_size, num_channel):
		self.metadata = metadata
		self.im_size = im_size
		self.im_shape = (im_size, im_size, num_channel)
		self.index = 0
		
	def next(self, batch_size=None, data_aug=False, random_shuffle=True):
		if batch_size is None:
			batch_size = len(self.metadata)

		# get the next batch_size rows from data
		if self.index + batch_size <= len(self.metadata):
			batch = self.metadata[self.index:self.index + batch_size]
			self.index += batch_size
		else:
			diff = batch_size - (len(self.metadata) - self.index)
			batch = self.metadata[self.index:] + self.metadata[:diff]
			self.index = diff

		# get the images and labels for the batch
		images = [None]*batch_size
		labels = [None]*batch_size
		rotations = [0, 90, 180, 270]
		flips = [True, False]
		for i in range(len(batch)):
			row = batch[i]
			images[i] = imresize(imread(row['path']), self.im_shape)
			images[i] = images[i][:, :, 0:3]

			if data_aug:
				images[i] = imrotate(images[i], np.random.choice(rotations))
				if np.random.choice(flips):
					images[i] = np.flipud(images[i])

			labels[i] = row['label']

		images = np.array(images)
		labels = np.array(labels)

		if random_shuffle:
			np.random.shuffle(self.metadata)

		return images, labels

	def data_in_batches(self, num_examples=None, batch_size=None, data_aug=False, random_shuffle=True):
		if num_examples is None:
			num_examples = len(self.metadata)
		num_examples = min(num_examples, len(self.metadata))

		if batch_size is None:
			batch_size = len(self.metadata)

		index = self.index
		self.index = 0
		                                                
		while self.index + batch_size <= num_examples:
			yield self.next(batch_size, data_aug, random_shuffle)

		self.index = index
