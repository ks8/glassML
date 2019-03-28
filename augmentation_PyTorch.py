# Perform on-the-fly data augmentation
import torch
import numpy as np


class Augmentation(object):

    def __init__(self, window_length=None):
        """
        Class for performing on-the-fly data augmentation.
        :param window_length: Length of window used to subsample data (window length of 1.0 implies no augmentation for
        normalized data)
        """
        self.window_length = window_length

    def __call__(self, data):

        window_lo = np.random.uniform(0.0, 1.0 - self.window_length)
        window_hi = window_lo + self.window_length

        pos, x = data.pos, data.x

        data_matrix = torch.cat((pos, x), 1)
        data_matrix = data_matrix[(data_matrix[:, 0] >= window_lo) & (data_matrix[:, 0] <= window_hi) &
                                  (data_matrix[:, 1] >= window_lo) & (data_matrix[:, 1] <= window_hi)]

        data.pos = data_matrix[:, 0:2]
        data.x = data_matrix[:, 2].unsqueeze(1)

        return data

    def __repr__(self):
        return '{}(cat={})'.format(self.__class__.__name__, self.window_length)


