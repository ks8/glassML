# Function for checking if there are any isolated nodes in a dataset
# Import modules
import argparse
from argparse import Namespace
from pprint import pformat

from compose_PyTorch import Compose
from distance_PyTorch import Distance
from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph


def isolated_nodes_check(args: Namespace):
    """
    Check if there are any isolated nodes in the specified dataset
    :param args: Dataset parameters
    """

    print(pformat(vars(args)))

    # Load data
    print('Loading data')
    data = GlassDataset(args.data_path, transform=Compose([NNGraph(args.num_neighbors), Distance(False)]))

    # Dataset length
    data_length = len(data)
    print('test size = {:,}'.format(
        data_length)
    )

    # Check for any graphs with isolated nodes
    for i in range(data_length):
        if data[i].contains_isolated_nodes():
            print(f'WARNING: Graph {i} in test set contains isolated nodes')
            exit(1)

    print('No isolated nodes discovered')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, dest='data_path', default=None,
                        help='.json metadata file location')
    parser.add_argument('--num_neighbors', type=int, dest='num_neighbors', default=1,
                        help='Number of nearest neighbors')

    args = parser.parse_args()

    isolated_nodes_check(args)


if __name__ == '__main__':
    main()
