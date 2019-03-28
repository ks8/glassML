# Average a set of RDFs
import argparse
from argparse import Namespace
import os
import numpy as np
import shutil
import scipy.stats


def combine_rdf(args: Namespace):
    """
    Average a set of RDFs contained in a folder. Note: some RDF files computed by LAMMPS have small printing errors so
    that the number of columns in the data is not always consistent for the last histogram bin. Avoid this by making
    max_rows < num RDF bins. TODO: Fix or avoid other similar errors
    :param args: args info
    """

    # data = None
    data = []
    num_files = 0

    for _, _, files in os.walk(args.data_dir):  # Loop through the RDF files in data_dir
        for f in files:
            # num_files += 1  # Count the number of RDFs being averaged
            shutil.copy(args.data_dir + '/' + f, '.')  # Copy the file into the working directory
            rdf_contents = open(f).readlines()
            rdf_contents = rdf_contents[args.skip_rows:args.skip_rows + args.max_rows]  # Read data subset
            os.remove(f)  # Remove the copied file from working directory
            new_file = open(f, 'w')
            for i in range(len(rdf_contents)):  # Write a new file that only contains the selected data subset
                new_file.write(rdf_contents[i])
            new_file.close()
            this_data = np.loadtxt(f)  # Load the data subset
            data.append(this_data)
            # if data is None:  # Combine with other RDF data
            #     data = this_data
            # else:
            #     data = data + this_data
            os.remove(f)

    data = np.array(data)

    rdf_mean = np.mean(data, axis=0)
    rdf_standard_error = scipy.stats.sem(data, axis=0)

    # data = data / float(num_files)  # Normalize

    np.savetxt(args.out_mean, rdf_mean, header='RDF mean values')  # Save data with out file name
    np.savetxt(args.out_sterr, rdf_standard_error, header='RDF standard error values')  # Save data with out file name


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing configs')
    parser.add_argument('--skip_rows', type=int, dest='skip_rows', default=4, help='Number of rows to skip')
    parser.add_argument('--max_rows', type=int, dest='max_rows', default=100, help='Nax number of rows to read')
    parser.add_argument('--out_mean', type=str, dest='out_mean', default=None, help='RDF mean output file name')
    parser.add_argument('--out_sterr', type=str, dest='out_sterr', default=None,
                        help='RDF standard error output file name')

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError('data_dir must be specified')

    if args.out_mean is None or args.out_sterr is None:
        raise ValueError('out_mean and out_sterr must be specified')

    if os.path.exists(args.out_mean):
        raise ValueError('out_mean file name already exists')

    if os.path.exists(args.out_sterr):
        raise ValueError('out_sterr file name already exists')

    combine_rdf(args)


if __name__ == "__main__":
    main()
