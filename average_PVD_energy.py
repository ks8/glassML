import argparse
import numpy as np
import os


def average_energy(args):
    """
    Compute average energy across trajectories of PVD glasses from DASH
    :param args: Energy file parameters
    """

    energies = []
    for root, subfolders, files in os.walk(args.data_dir):
        for f in files:
            if f[0:3] == 'EIS':
                data = np.loadtxt(args.data_dir + '/' + f)
                energies.append(data)

    energies = np.array(energies)
    print('Mean EIS: ', np.mean(energies))
    print('Std EIS: ', np.std(energies))


def main():
    """
    Parse arguments and execute xyz file processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_dir', type=str, dest='data_dir', default=None, help='Directory containing energy files')

    args = parser.parse_args()
    average_energy(args)


if __name__ == "__main__":
    main()
