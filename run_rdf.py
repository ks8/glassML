# Compute and average RDFs for a set of configurations
import argparse
import os
import numpy as np
import shutil


def run_rdf(args):
    """
    Compute RDFs for a set of configurations.
    :param args: args info
    """

    os.makedirs(args.out_dir, exist_ok=True)

    for _, _, files in os.walk(args.data_dir):
        for f in files:
            data = np.loadtxt(args.data_dir + '/' + f, skiprows=2)
            num_atoms = len(data)
            system_contents = open(args.sys_file).readlines()
            system_contents[2] = '          ' + str(num_atoms) + ' atoms' + '\n'
            new_file = open(args.sys_file, 'w')
            for i in range(len(system_contents)):
                new_file.write(system_contents[i])
            new_file.close()
            shutil.copy(args.data_dir + '/' + f, '.')
            os.system('./get_rdfs' + ' ' + '1' + ' ' + f + ' ' + args.sys_file + ' ' + '<' + ' ' + args.input_file)
            shutil.copy('1_1.rdf', args.out_dir + '/' + '1_1-RDF-' + f)
            shutil.copy('1_2.rdf', args.out_dir + '/' + '1_2-RDF-' + f)
            shutil.copy('2_2.rdf', args.out_dir + '/' + '2_2-RDF-' + f)
            os.remove(f)
            os.remove('1_1.rdf')
            os.remove('1_2.rdf')
            os.remove('2_2.rdf')


def main():
    """
    Parse arguments and execute metadata creation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing configs')
    parser.add_argument('--sys_file', type=str, dest='sys_file', default=None, help='File containing sys info')
    parser.add_argument('--input_file', type=str, dest='input_file', default=None, help='rdf input file')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None, help='Directory for output files')

    args = parser.parse_args()
    run_rdf(args)


if __name__ == "__main__":
    main()
