# Normalize trajectories for PVD and LC glass dataset
import json
import os
import re
import argparse
import numpy as np


def process_metadata(args):
    """
    Create metadata folder and file.
    :param args: Folder name info
    """

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for root, subfolders, files in os.walk(args.data_dir):
        pvd_num = args.pvd_label_start  # Relabeling PVD examples with uid
        for f in files:
            path = os.path.join(root, f)
            category = re.findall(r'glass', path)
            if len(category) == 1:
                data = np.loadtxt(path, skiprows=0)[:, 0:3]  # TODO: Make sure density stuff is the same - check average density of PVD films
                data = data[(data[:, 1] <= np.sqrt(float(250)/float(4320))) &
                            (data[:, 2] <= np.sqrt(float(250)/float(4320)))]
            else:
                data = np.loadtxt(path, skiprows=2)[:, 0:3]
                data = data[(data[:, 0] > 0.0) & (data[:, 1] >= 10.0) & (data[:, 1] <= 25.0) & (data[:, 2] >= 10.0) &
                            (data[:, 2] <= 25.0)]  # Selecting for atoms in the bulk region
            data[:, 1] = data[:, 1] - np.min(data[:, 1])
            data[:, 2] = data[:, 2] - np.min(data[:, 2])

            for i in range(2):
                data[:, i + 1] = data[:, i + 1] / (np.max(data[:, i + 1]))  # Normalizing

            if str(f)[-4:] == '.xyz':
                new_file = open(args.out_dir + '/' + str(f)[:-4] + '--' + str(pvd_num)
                                + '_normalized' + str(f)[-4:], 'w')
                pvd_num += 1
            else:
                new_file = open(args.out_dir + '/' + str(f)[:-4] + '_normalized' + str(f)[-4:], 'w')
            for i in range(len(data)):
                new_file.write(str(data[i, 0]) + '   ' + str(data[i, 1]) + '   ' + str(data[i, 2]) + '\n')


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing data')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory for output data')
    parser.add_argument('--pvd_label_start', type=int, dest='pvd_label_start', default=0, help='PVD labeling')

    args = parser.parse_args()
    process_metadata(args)


if __name__ == "__main__":
    main()
