# Function for computing mean and standard deviation of energies across a restricted set of trajectories
import argparse
import numpy as np
import os
import shutil


def clip_energy(args):
    """
    Compute average and standard deviation of energy across trajectories of glasses from DASH within a specified
    energy range and create a new folder containing trajectories that are within this range. The target data_dir must
    contain both trajectory files and inherent structural energy files.
    :param args: Energy file parameters
    """

    energies = []  # List of inherent structural energy values for configurations within range
    valid_files = []  # List of file names for configurations with energies within range
    for root, subfolders, files in os.walk(args.data_dir):
        for f in files:
            if f[0:3] == 'EIS':
                data = np.loadtxt(args.data_dir + '/' + f)  # Load EIS (inherent structural energy) file
                if args.energy_lo <= data <= args.energy_hi:
                    energies.append(data)
                    valid_files.append(f)

    energies = np.array(energies)
    print('Mean EIS: ', np.mean(energies))
    print('Std EIS: ', np.std(energies))
    print(energies.shape)

    os.makedirs(args.out_dir, exist_ok=False)  # Make the out_dir if it does not already exist
    for f in valid_files:
        shutil.copy(args.data_dir + '/' + 'final_' + f[f.find('EIS') + 4:-4] + '.xyz', args.out_dir)


def main():
    """
    Parse arguments and execute xyz file processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, dest='data_dir', default=None, help='Directory containing energy files')
    parser.add_argument('--energy_lo', type=float, dest='energy_lo', default=None, help='Low energy threshold')
    parser.add_argument('--energy_hi', type=float, dest='energy_hi', default=None, help='High energy threshold')
    parser.add_argument('--out_dir', type=str, dest='out_dir', default=None,
                        help='Directory containing output files')

    args = parser.parse_args()
    clip_energy(args)


if __name__ == "__main__":
    main()
