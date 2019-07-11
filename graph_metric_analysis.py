# Main function for graph metric analysis
from argparse import Namespace
import json
from tqdm import tqdm
import os
import numpy as np

from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader
from distance_PyTorch import Distance
from GlassDataset_PyTorch import GlassDataset
from nngraph_PyTorch import NNGraph
from augmentation_PyTorch import Augmentation

from GlassBatchMolGraph import GlassBatchMolGraph

from parsing import parse_train_args

import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def run_graph_metric(args: Namespace):
    """
    Evaluates a dataset of configurations based on attention graph metrics
    :param args: Set of args
    """

    # Load metadata
    metadata = json.load(open(args.data_path, 'r'))

    # Train/val/test split
    if args.k_fold_split:
        data_splits = []
        kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
        for train_index, test_index in kf.split(metadata):
            splits = [train_index, test_index]
            data_splits.append(splits)
        data_splits = data_splits[args.fold_index]

        if args.use_inner_test:
            train_indices, remaining_indices = train_test_split(data_splits[0], test_size=args.val_test_size,
                                                                random_state=args.seed)
            validation_indices, test_indices = train_test_split(remaining_indices, test_size=0.5,
                                                                random_state=args.seed)

        else:
            train_indices = data_splits[0]
            validation_indices, test_indices = train_test_split(data_splits[1], test_size=0.5, random_state=args.seed)

        train_metadata = list(np.asarray(metadata)[list(train_indices)])
        validation_metadata = list(np.asarray(metadata)[list(validation_indices)])
        test_metadata = list(np.asarray(metadata)[list(test_indices)])

    else:
        train_metadata, remaining_metadata = train_test_split(metadata, test_size=args.val_test_size,
                                                              random_state=args.seed)
        validation_metadata, test_metadata = train_test_split(remaining_metadata, test_size=0.5, random_state=args.seed)

    # Load dataset
    transform = Compose([Augmentation(args.augmentation_length), NNGraph(args.num_neighbors), Distance(False)])
    data = GlassDataset(test_metadata, transform=transform)
    args.atom_fdim = 3
    args.bond_fdim = args.atom_fdim + 1

    # Dataset length
    data_length = len(data)
    print('data size = {:,} '.format(data_length))

    # Convert to iterator
    data = DataLoader(data, args.batch_size)

    # Run through dataset in batches
    for batch in tqdm(data, total=len(data)):

        # Prepare batch
        mol_graph = GlassBatchMolGraph(batch)

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, uid, targets = mol_graph.get_components()

        # Loop through the individual graphs in the batch
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:  # Skip over empty graphs
                continue
            else:
                atoms = f_atoms[a_start:a_start + a_size, :].cpu().numpy()  # Atom features
                bonds1 = a2b[a_start:a_start + a_size, :]  # a2b
                bonds2 = b2a[b_scope[i][0]:b_scope[i][0] + b_scope[i][1]]  # b2a
                bonds_dict = {}  # Dictionary to keep track of atoms that bonds are connecting
                for k in range(len(bonds2)):  # Collect info from b2a
                    bonds_dict[k + 1] = bonds2[k].item() - a_start + 1
                for k in range(bonds1.shape[0]):  # Collect info from a2b
                    for m in range(bonds1.shape[1]):
                        bond_num = bonds1[k, m].item() - b_scope[i][0] + 1
                        if bond_num > 0:
                            bonds_dict[bond_num] = (bonds_dict[bond_num], k + 1)

                # Save weights for this graph
                label = targets[i].item()  # Graph label

                # Graph analysis
                graph = nx.Graph()  # Construct an empty graph for high attention weights
                graph_lo = nx.Graph()  # Construct an empty graph for low attention weights
                num_type_2_connections = 0  # Initialize number of type 2 bonds counter
                num_type_1_isolated = 0  # Initialize num type 1 isolated
                num_type_2_isolated = 0  # Initialize num type 2 isolated

                for j in range(len(bonds_dict)):
                    # Analysis
                    if atoms[bonds_dict[j + 1][0] - 1, 2] == 2.0 and atoms[bonds_dict[j + 1][1] - 1, 2] == 2.0 and \
                            (bonds_dict[j + 1][0], bonds_dict[j + 1][1]) not in graph.edges:  # Num type 2 bonds
                        num_type_2_connections += 1
                    if atoms[bonds_dict[j + 1][0] - 1, 2] == 2.0 or atoms[bonds_dict[j + 1][1] - 1, 2] == 2.0:
                        graph.add_nodes_from([bonds_dict[j + 1][0], bonds_dict[j + 1][1]])  # Record nodes for this edge
                        graph.add_edge(bonds_dict[j + 1][0], bonds_dict[j + 1][1])  # Record the edge

                for j in range(len(bonds_dict)):
                    for k in range(2):
                        if bonds_dict[j + 1][k] not in graph:
                            if bonds_dict[j + 1][k] not in graph_lo:
                                if atoms[bonds_dict[j + 1][k] - 1, 2] == 1.0:
                                    num_type_1_isolated += 1
                                else:
                                    num_type_2_isolated += 1
                                graph_lo.add_node(bonds_dict[j + 1][k])

                # Write analysis results
                if not os.path.exists(args.save_dir + '/' + 'attention_analysis.txt'):
                    f = open(args.save_dir + '/' + 'attention_analysis.txt', 'w')
                    f.write('# Category    Subgraphs    Type 2 Bonds    Type 1 Isolated    '
                            'Type 2 Isolated' + '\n')
                else:
                    f = open(args.save_dir + '/' + 'attention_analysis.txt', 'a')
                f.write(str(label) + '    ' + str(nx.number_connected_components(graph)) + '    ' +
                        str(num_type_2_connections) + '    ' + str(num_type_1_isolated) + '    ' +
                        str(num_type_2_isolated) + '\n')


if __name__ == '__main__':
    args = parse_train_args()
    args.num_tasks = 1
    run_graph_metric(args)

