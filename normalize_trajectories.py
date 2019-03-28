# Normalize DASH trajectories
import os
import argparse
import numpy as np


def normalize_configs(args):
    """
    Normalize DASH configurations.
    :param args: args info
    """

    os.makedirs(args.out_dir, exist_ok=False)  # Make the out_dir if it does not already exist

    for root, subfolders, files in os.walk(args.data_dir):
        config_num = args.config_label_start  # Relabeling examples with uid
        for f in files:
            path = os.path.join(root, f)  # File path

            # Read a single configuration skipping args.skip_rows
            contents = open(path).readlines()
            num_atoms = int(contents[0].split()[0])
            new_file = open(args.out_dir + '/' + f, 'w')
            for i in range(num_atoms + args.skip_rows):
                new_file.write(contents[i])
            new_file.close()
            data = np.loadtxt(args.out_dir + '/' + f, skiprows=args.skip_rows)[:, 0:3]  # Load the data
            os.remove(args.out_dir + '/' + f)  # Remove temp file

            # Select for atoms in a specified y-range
            data = data[(data[:, 0] > 0.0) & (data[:, 2] >= args.y_lo) & (data[:, 2] <= args.y_hi)]

            # Normalize positions
            data[:, 1] = data[:, 1] - np.min(data[:, 1])  # Subtract out the minimum x, y values so that min x, y are 0
            data[:, 2] = data[:, 2] - np.min(data[:, 2])
            for i in range(2):
                data[:, i + 1] = data[:, i + 1] / (np.max(data[:, i + 1]))  # Divide by max x, y values

            # Write new file to args.out_dir
            new_file = open(args.out_dir + '/' + str(f)[:-4] + '--' + str(config_num)
                            + '_normalized' + str(f)[-4:], 'w')
            config_num += 1
            for i in range(len(data)):
                new_file.write(str(data[i, 0]) + '   ' + str(data[i, 1]) + '   ' + str(data[i, 2]) + '\n')


def main():
    """
    Parse arguments and execute normalization.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing data')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory for output data')
    parser.add_argument('--config_label_start', type=int, dest='config_label_start', default=0, help='Config labeling')
    parser.add_argument('--skip_rows', type=int, dest='skip_rows', default=2, help='Number of rows to skip')
    parser.add_argument('--y_lo', type=float, dest='y_lo', default=10.0, help='Low value in y-dimension')
    parser.add_argument('--y_hi', type=float, dest='y_hi', default=25.0, help='High value in y-dimension')

    args = parser.parse_args()

    normalize_configs(args)


if __name__ == "__main__":
    main()
