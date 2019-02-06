import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def plot_cooling(args):
    """
    Plot two cooling curves simultaneously.
    :param args: Arguments passed from main().
    """

    data_1 = np.loadtxt(args.data_1)
    data_2 = np.loadtxt(args.data_2)

    x = np.linspace(1.999025, 0.05, 200)[20:]

    plt.plot(x, data_1[20:, 1], color='darkred', label='2e5')
    plt.plot(x, data_2[20:, 1], color='b', label='2e7')
    plt.hlines(-3.86744, 0.05, 1.75, linestyle='dashed', label='Dataset energy')
    plt.ylabel('Inherent Structural Energy (LJ units)')
    plt.xlabel('Temperature (LJ units)')
    plt.legend()

    plt.title(args.plot_title)

    plt.savefig(args.output)


def main():
    """
    Parse arguments and execute plotting.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data_1', type=str, dest='data_1', default='', help='File name of first cooling curve')
    parser.add_argument('-data_2', type=str, dest='data_2', default='', help='File name of second cooling curve')
    parser.add_argument('-plot_title', type=str, dest='plot_title', default='Density Profile', help='Plot title')
    parser.add_argument('-output', type=str, dest='output', default=None, help='File name for output files')

    args = parser.parse_args()

    plot_cooling(args)


if __name__ == "__main__":
    main()