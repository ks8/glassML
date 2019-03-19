# Average a set of RDFs
import argparse
import os
import numpy as np


def run_rdf(args):
    """
    Average RDFs.
    :param args: args info
    """

    data_1_1 = None
    data_1_2 = None
    data_2_2 = None

    num_1_1_files = 0
    num_1_2_files = 0
    num_2_2_files = 0

    for _, _, files in os.walk(args.data_dir):
        for f in files:
            if f[0:3] == '1_1':
                num_1_1_files += 1
                if data_1_1 is None:
                    data_1_1 = np.loadtxt(args.data_dir + '/' + f)
                else:
                    data_1_1 = data_1_1 + np.loadtxt(args.data_dir + '/' + f)

            if f[0:3] == '1_2':
                num_1_2_files += 1
                if data_1_2 is None:
                    data_1_2 = np.loadtxt(args.data_dir + '/' + f)
                else:
                    data_1_2 = data_1_2 + np.loadtxt(args.data_dir + '/' + f)

            if f[0:3] == '2_2':
                num_2_2_files += 1
                if data_2_2 is None:
                    data_2_2 = np.loadtxt(args.data_dir + '/' + f)
                else:
                    data_2_2 = data_2_2 + np.loadtxt(args.data_dir + '/' + f)

    data_1_1 = data_1_1 / num_1_1_files
    data_1_2 = data_1_2 / num_1_2_files
    data_2_2 = data_2_2 / num_2_2_files

    np.savetxt('1_1.rdf', data_1_1)
    np.savetxt('1_2.rdf', data_1_2)
    np.savetxt('2_2.rdf', data_2_2)


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing configs')

    args = parser.parse_args()
    run_rdf(args)


if __name__ == "__main__":
    main()
