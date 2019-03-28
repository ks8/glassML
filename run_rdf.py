# Compute 1-1, 1-2, and 2-2 RDFs for a set of 2D atomic configurations that have been prepared as LAMMPS input files
import argparse
from argparse import Namespace
import os
import shutil


def run_rdf(args: Namespace):
    """
    Compute RDFs for a set of configurations that have been prepared as LAMMPS input files.
    :param args: args info
    """

    os.makedirs(args.out_dir, exist_ok=False)  # Make output directory

    for _, _, files in os.walk(args.data_dir):  # Loop through files in data_dir
        for f in files:
            shutil.copy(args.data_dir + '/' + f, '.')  # Copy the file into the working directory
            os.system('mv ' + f + ' ' + args.input_file_name)  # Rename the file

            # Modify the LAMMPS run script to output rdf file in specified directory and with modified file name
            make_contents = open(args.make_file_name).readlines()
            make_contents[-4] = 'fix rdfout all ave/time 1 1 1 c_calcrdf[1] c_calcrdf[2] c_calcrdf[4] c_calcrdf[6] ' \
                                'mode vector file ' + args.out_dir + '/' + 'RDF-' + f + '\n'
            new_file = open(args.make_file_name, 'w')
            for i in range(len(make_contents)):
                new_file.write(make_contents[i])
            new_file.close()

            # Run the batch script
            os.system('sbatch ' + args.batch_script_name)


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing configs')
    parser.add_argument('--input_file_name', type=str, dest='input_file_name', default='in.config',
                        help='LAMMPS input file name')
    parser.add_argument('--make_file_name', type=str, dest='make_file_name', default='in.make-glass',
                        help='LAMMPS script name')
    parser.add_argument('--batch_script_name', type=str, dest='batch_script_name', default='submit.sbatch',
                        help='LAMMPS script name')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory for output files')

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError('data_dir must be specified')

    if args.out_dir is None:
        raise ValueError('out_dir must be specified')

    run_rdf(args)


if __name__ == "__main__":
    main()
