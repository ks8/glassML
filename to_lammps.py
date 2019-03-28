# Convert 2D atomic coordinates to LAMMPS input file in order to process RDF using LAMMPS
import os
import argparse
from argparse import Namespace
import numpy as np


def to_lammps(args: Namespace):
    """
    Convert atomic coordinates to LAMMPS input file.
    :param args: args info
    """

    os.makedirs(args.out_dir, exist_ok=False)  # Make output directory (only if not already exist)

    for _, _, files in os.walk(args.data_dir):  # Loop through files in data_dir to process each one
        for f in files:
            if args.max_rows:
                contents = open(args.data_dir + '/' + f).readlines()
                num_atoms = int(contents[0].split()[0])
                new_file = open(args.out_dir + '/' + f, 'w')
                for i in range(num_atoms + args.skip_rows):
                    new_file.write(contents[i])
                new_file.close()
                data = np.loadtxt(args.out_dir + '/' + f, skiprows=args.skip_rows)  # Load the data
                os.remove(args.out_dir + '/' + f)
            else:
                data = np.loadtxt(args.data_dir + '/' + f, skiprows=args.skip_rows)  # Load the data

            # Select a subset of the data if specified within the bounds (x_lo, y_lo) and (x_hi, y_hi)
            if args.subset_data:
                data = data[(data[:, 1] >= args.x_lo) & (data[:, 1] <= args.x_hi) &
                            (data[:, 2] >= args.y_lo) & (data[:, 2] <= args.y_hi)]

            # Write new LAMMPS file
            new_file = open(args.out_dir + '/' + str(f[:-4]) + '-RDF-prepared.txt', 'w')
            new_file.write('LAMMPS Description' + '\n')
            new_file.write('\n')
            new_file.write(str(len(data)) + '        ' + 'atoms' + '\n')  # Num atoms
            new_file.write('\n')
            new_file.write(str(args.num_atom_types) + '        atom types' + '\n')  # Num atom types
            new_file.write('\n')
            new_file.write(str(np.min(data[:, 1] - 1e-5)) + ' ' + str(np.max(data[:, 1] + 1e-5))
                           + ' ' + 'xlo xhi' + '\n')  # Add x bounds (with some padding)
            new_file.write('\n')
            new_file.write(str(np.min(data[:, 2] - 1e-5)) + ' ' + str(np.max(data[:, 2] + 1e-5))
                           + ' ' + 'ylo yhi' + '\n')  # Add y bounds (with some padding)
            new_file.write('\n')
            new_file.write(str(-3.0) + ' ' + str(3.0) + ' ' + 'zlo zhi' + '\n')  # Add z bounds for 2D simulation
            new_file.write('\n')
            new_file.write('Atoms' + '\n')  # Atoms section
            new_file.write('\n')

            for i in range(len(data)):  # Write atom numbers, types, and positions to file
                new_file.write(str(i + 1) + '  ' + str(int(data[i, 0])) + '  ' + str(data[i, 1]) + '  ' +
                               str(data[i, 2]) + '  ' + str(0.0) + '\n')


def main():
    """
    Parse arguments and execute LAMMPS file creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory of config files')
    parser.add_argument('--skip_rows', type=int, dest='skip_rows', default=0, help='Number of rows to skip in data')
    parser.add_argument('--max_rows', action='store_true', default=False, help='Restrict max rows as num atoms')
    parser.add_argument('--x_lo', type=float, dest='x_lo', default=None, help='x_lo bound for coordinates to process')
    parser.add_argument('--x_hi', type=float, dest='x_hi', default=None, help='x_hi bound for coordinates to process')
    parser.add_argument('--y_lo', type=float, dest='y_lo', default=None, help='y_lo bound for coordinates to process')
    parser.add_argument('--y_hi', type=float, dest='y_hi', default=None, help='y_hi bound for coordinates to process')
    parser.add_argument('--num_atom_types', type=int, dest='num_atom_types', default=2,
                        help='Number of atom types in coordinate file')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Output file name')

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError('data_dir must be specified')

    if args.out_dir is None:
        raise ValueError('out_dir must be specified')

    if args.x_lo is not None or args.x_hi is not None or args.y_lo is not None or args.y_hi is not None:
        assert(args.x_lo is not None and args.x_hi is not None and args.y_lo is not None and args.y_hi is not None)
        args.subset_data = True
    else:
        args.subset_data = False

    to_lammps(args)


if __name__ == "__main__":
    main()
