# Visualize a pair of RDFs, where each file has 5 columns corresponding to index, r, g_11(r), g_12(r), and g_22(r)
import argparse
from argparse import Namespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def visualize_rdf(args: Namespace):
    """
    Visualize a pair of RDFs with standard errors.
    :param args: args info
    """

    # Load data
    data_1 = np.loadtxt(args.data_1)
    data_err_1 = np.loadtxt(args.data_err_1)
    data_2 = np.loadtxt(args.data_2)
    data_err_2 = np.loadtxt(args.data_err_2)

    # Plot
    plt.errorbar(x=data_1[:, 1], y=data_1[:, 4], yerr=2*data_err_1[:, 4], color='b', ecolor='lightblue',
                 label=r'$t_{dep}=1\times10^5, \hspace{0.2} Tsub = 0.18$')
    plt.errorbar(x=data_2[:, 1], y=data_2[:, 4], yerr=2*data_err_2[:, 4], color='g', ecolor='lightgreen',
                 label=r'$t_{cool}=4.7\times10^3, \hspace{0.2} T = 0.05$')

    plt.xlabel(r'$r \hspace{.5} / \hspace{.5} \sigma$', size=15)
    plt.ylabel(r'$g_{22}(r)$', size=15)
    plt.legend()
    plt.savefig(args.out)
    plt.clf()


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_1', type=str, dest='data_1', default=None, help='First RDF file')
    parser.add_argument('--data_err_1', type=str, dest='data_err_1', default=None, help='First errors file')
    parser.add_argument('--data_2', type=str, dest='data_2', default=None, help='Second RDF file')
    parser.add_argument('--data_err_2', type=str, dest='data_err_2', default=None, help='Second errors file')
    parser.add_argument('--out', type=str, dest='out', default=None, help='Output file name')

    args = parser.parse_args()

    if args.data_1 is None or args.data_2 is None:
        raise ValueError('data_1 and data_2 must be specified')

    if args.out is None:
        raise ValueError('out file name must be specified')

    if os.path.exists(args.out):
        raise ValueError('out file name already exists')

    visualize_rdf(args)


if __name__ == "__main__":
    main()
