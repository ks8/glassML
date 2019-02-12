import argparse
import numpy as np


def average_energy(args):
    """
    Compute average energy across trajectories
    :param args: Energy file parameters
    """

    total_data = np.zeros((args.num_trajectories, args.num_steps, 2))

    for i in range(args.traj_file_start, args.traj_file_start + args.num_trajectories):
        print(i)
        data = np.loadtxt(args.data + '/' + str(i) + '/' + 'cooling.dat')
        total_data[i-1] = data

    total_data_mean = np.mean(total_data, axis=0)
    total_data_std = np.std(total_data, axis=0)

    f = open(args.output + '.txt', 'w')
    f.write('# ' + 'Timestep' + '    ' + 'Mean_Energy' + '    ' + 'SD_Energy' + '\n')
    for i in range(total_data_mean.shape[0]):
        f.write(str(total_data_mean[i][0]) + '    ' + str(total_data_mean[i][1]) + '    ' + str(total_data_std[i][1])
                + '\n')


def main():
    """
    Parse arguments and execute xyz file processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, dest='data', default=None, help='Directory containing trajectories')
    parser.add_argument('-output', type=str, dest='output', default=None, help='File name for output files')
    parser.add_argument('-traj_file_start', type=int, dest='traj_file_start', default=0,
                        help='Initial directory name for trajectory files: usually begins with 0 or 1')
    parser.add_argument('-num_trajectories', type=int, dest='num_trajectories', default=10000,
                        help='Number of energy files to analyze')
    parser.add_argument('-num_steps', type=int, dest='num_steps', default=200,
                        help='Number of energy steps to analyze')

    args = parser.parse_args()
    average_energy(args)


if __name__ == "__main__":
    main()
