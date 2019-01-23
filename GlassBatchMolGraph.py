# Class that transforms a batch of GlassDataset data into a batch of data compatible with the chemprop repository
# Import modules
from typing import List, Tuple
import torch
import numpy as np


class GlassBatchMolGraph:

    def __init__(self, example):
        """
        Class that transforms a batch of GlassDataset data into a batch of data compatible with the chemprop repo
        :param example:
        """
        self.batch_size = len(torch.unique(example.batch))  # Extract the batch size
        self.atom_fdim = example.pos.size(1) + example.x.size(1)  # Dimensionality of atom features (pos + atom type)
        bond_fdim = example.edge_attr.size(1)  # Dimensionality of bond features (distance measure)
        self.bond_fdim = self.atom_fdim + bond_fdim  # Total bond dimension (given as atom + bond dimensionality)
        self.n_atoms = 1  # Number of atoms (+1 for padding)
        self.n_bonds = 1  # Number of bonds (+1 for padding)
        self.a_scope = []  # List of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # List of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.f_atoms = [torch.zeros(self.atom_fdim)]  # List of atom features (including one torch.zeros for padding)
        self.f_bonds = [torch.zeros(self.bond_fdim)]  # List of total bond features (including padding)
        self.a2b = [[]]  # Maps from an atom index to a list of indices of all bonds that point to that atom
        self.b2a = [0]  # Maps from a bond index to the atom index such that that bond comes from that atom
        self.b2revb = [0]  # Maps from a bond index to the index of the reverse bond (bond i --> j maps to bond j -- > )

        # Atom information
        count = 0
        current_example_index = 0
        for example_index, pos, x in zip(example.batch, example.pos, example.x):
            self.f_atoms.append(torch.cat((pos, x), dim=0))

            if example_index != current_example_index:  # If we have reached a new molecule
                start = self.n_atoms - count
                self.a_scope.append((start, count))  # Record a_scope index where molecule starts and its # of atoms
                current_example_index = example_index
                count = 0

            count += 1
            self.n_atoms += 1
            self.a2b.append([])  # Append an empty list to a2b for every atom in the batch

        start = self.n_atoms - count
        self.a_scope.append((start, count))

        # Bond information
        count = 0  # Count of bonds we have added in a given molecule
        current_example_index = 0  # Keep track of which molecule we are extracting from
        reduced_edge_index, reduced_indices = np.unique(np.array(
            [tuple(np.sort(p)) for p in example.edge_index.t()]), axis=0, return_index=True)
        reduced_edge_index = torch.from_numpy(reduced_edge_index)  # Edge index with unique bond information
        reduced_edge_attr = example.edge_attr[reduced_indices]  # Edge attr with unique bond information

        for (a1, a2), edge_attr in zip(reduced_edge_index, reduced_edge_attr):
            a1, a2 = a1.item(), a2.item()  # Atoms bonded to each other
            example_index = example.batch[a1]  # Which molecule this bond belongs to

            a1, a2 = a1 + 1, a2 + 1  # +1 for padding

            self.f_bonds.append(torch.cat((self.f_atoms[a1], edge_attr), dim=0))  # Features of bond a1 --> a2
            self.f_bonds.append(torch.cat((self.f_atoms[a2], edge_attr), dim=0))  # Features of bond a2 --> a1

            b1 = self.n_bonds  # b1 = a1 --> a2
            b2 = b1 + 1  # b2 = a2 --> a1

            self.a2b[a2].append(b1)
            self.a2b[a1].append(b2)

            self.b2a.append(a1)
            self.b2a.append(a2)

            self.b2revb.append(b2)
            self.b2revb.append(b1)

            if example_index != current_example_index:
                start = self.n_bonds - count
                self.b_scope.append((start, count))
                current_example_index = example_index
                count = 0

            count += 2  # 2 because we encode directed edges
            self.n_bonds += 2

        start = self.n_bonds - count
        self.b_scope.append((start, count))

        # Cast to tensor
        self.max_num_bonds = max(len(in_bonds) for in_bonds in self.a2b)  # Zero padding for a2b
        self.f_atoms = torch.stack(self.f_atoms)
        self.f_bonds = torch.stack(self.f_bonds)
        self.a2b = torch.LongTensor([self.a2b[a] +
                                     [0] * (self.max_num_bonds - len(self.a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(self.b2a)
        self.b2revb = torch.LongTensor(self.b2revb)

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def __len__(self) -> int:
        return self.batch_size
