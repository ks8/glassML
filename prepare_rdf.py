# Prepare a configuration for RDF analysis
import argparse
import numpy as np
import os


def prepare_rdf(args):
    """
    Prepare a directory of configurations for RDF analysis.
    :param args: args info
    """

    os.makedirs(args.out_dir, exist_ok=True)

    for _, _, files in os.walk(args.data_dir):
        for f in files:
            new_file = open(args.out_dir + '/' + f[:-4] + '-RDF' + '.txt', 'w')
            data = np.loadtxt(args.data_dir + '/' + f)
            new_file.write(str(len(data)) + '\n')
            new_file.write('bounds lo (0.0, 0.0, 0.0) hi (1.0, 1.0, 1.0)' + '\n')
            for i in range(len(data)):
                new_file.write(str(int(data[i, 0])) + '    ' + str(data[i, 1]) + '    ' + str(data[i, 2]) +
                               '    ' + str(0.0) + '\n')
            new_file.close()


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing configs')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory for output files')

    args = parser.parse_args()
    prepare_rdf(args)


if __name__ == "__main__":
    main()
