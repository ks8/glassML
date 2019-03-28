# Convert atomic coordinates to LAMMPS input file in order to process RDF
import os
import argparse
import numpy as np


def to_lammps(args):
    """
    Convert atomic coordinates to LAMMPS input file
    :param args: args info
    """

    if not os.path.exists(args.out_dir):  # Make output directory if it does not already exist
        os.makedirs(args.out_dir)

    for _, _, files in os.walk(args.data_dir):  # Loop through files in data_dir to process each one
        for f in files:
            data = np.loadtxt(args.data_dir + '/' + f, skiprows=args.skip_rows)
            data = data[(data[:, 0] > 0.0) & (data[:, 1] >= args.x_lo) & (data[:, 1] <= args.x_hi) &
                        (data[:, 2] >= args.y_lo) & (data[:, 2] <= args.y_hi)]

            new_file = open(args.out_dir + '/' + f[:-4] + '-RDF.txt', 'w')
            new_file.write('LAMMPS Description' + '\n')
            new_file.write('\n')
            new_file.write(str(len(data)) + '        ' + 'atoms' + '\n')
            new_file.write('\n')
            new_file.write('2        atom types' + '\n')
            new_file.write('\n')
            new_file.write(args.x_lo + ' ' + args.x_hi + ' ' + 'xlo xhi' + '\n')
            new_file.write('\n')
            new_file.write(args.y_lo + ' ' + args.y_hi + ' ' + 'xlo xhi' + '\n')
            new_file.write('\n')
            new_file.write(args.z_lo + ' ' + args.z_hi + ' ' + 'xlo xhi' + '\n')
            new_file.write('\n')
            new_file.write('Atoms' + '\n')
            new_file.write('\n')

            for i in range(len(data)):
                new_file.write(str(i + 1) + '  ' + str(int(data[i, 0])) + '  ' + str(data[i, 1]) + '  ' +
                               str(data[i, 2]) + '  ' + str(0.0) + '\n')


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory of config files')
    parser.add_argument('--skip_rows', type=int, dest='skip_rows', default=0, help='Number of rows to skip in data')
    parser.add_argument('--x_lo', type=float, dest='x_lo', default=0.0, help='x_lo bound for coordinates to process')
    parser.add_argument('--x_hi', type=float, dest='x_hi', default=1.0, help='x_hi bound for coordinates to process')
    parser.add_argument('--y_lo', type=float, dest='y_lo', default=0.0, help='y_lo bound for coordinates to process')
    parser.add_argument('--y_hi', type=float, dest='y_hi', default=1.0, help='y_hi bound for coordinates to process')
    parser.add_argument('--z_lo', type=float, dest='z_lo', default=-3.0, help='y_lo bound for coordinates to process')
    parser.add_argument('--z_hi', type=float, dest='z_hi', default=3.0, help='y_hi bound for coordinates to process')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Output file name')

    args = parser.parse_args()
    to_lammps(args)


if __name__ == "__main__":
    main()
