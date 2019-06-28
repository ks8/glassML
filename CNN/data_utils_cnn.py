# Auxiliary data processing functions for CNN
from typing import List, Dict
import numpy as np


def get_unique_labels(metadata) -> List:
    """
    Gets unique labels in a metadata file.
    :param metadata: Metadata JSON file
    :return: List of unique labels
    """
    return list(set([row['label'] for row in metadata]))


def create_one_hot_mapping() -> Dict:
    """
    Creates a one-hot mapping for glass and liquid labels.
    :return: One-hot mapping dictionary
    """
    one_hot_mapping = dict()

    one_hot = np.zeros(2)
    one_hot[0] = 1
    one_hot_mapping['liquid'] = one_hot

    one_hot = np.zeros(2)
    one_hot[1] = 1
    one_hot_mapping['glass'] = one_hot

    return one_hot_mapping


def convert_to_one_hot(metadata, one_hot_mapping: Dict):
    """
    Converts labels to one-hot vectors.
    :param metadata: Metadata JSON file
    :param one_hot_mapping: One-hot mapping dictionary
    :return: Metadata with one-hot vector labels
    """
    for row in metadata:
        row['original_label'] = row['label']
        row['label'] = one_hot_mapping[row['label']]

    return metadata

