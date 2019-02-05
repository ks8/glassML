import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def process_xyz(args):
    """
    Process and plot density and composition information from PVD trajectory files
    :param args: Arguments passed from main()
    """

    contents_traj = open(args.data, 'r')

    still_data = True
    hist = None
    hist_fractions = None
    bounds_ylo = None
    bounds_yhi = None

    while still_data:

        line = contents_traj.readline()
        if len(line) > 0:
            num_atoms = int(line)
            line = contents_traj.readline().split()
            bounds_ylo = float(line[3][0:(len(line[3]) - 1)])
            bounds_yhi = float(line[7][0:(len(line[7]) - 1)])

            positions = []
            types = []
            for i in range(num_atoms):
                atom = contents_traj.readline().split()
                positions.append([float(atom[1]), float(atom[2]), float(atom[3])])
                types.append(int(atom[0]))
            positions = np.array(positions)
            types = np.array(types)
            y_positions = positions[types != 0, 1]
            type_one_y_positions = positions[types == 1, 1]

            hist, _ = np.histogram(y_positions, range=(bounds_ylo, bounds_yhi), bins=args.num_bins)
            hist_type_one, _ = np.histogram(type_one_y_positions, range=(bounds_ylo, bounds_yhi), bins=args.num_bins)

            hist_fractions = np.nan_to_num(np.divide(np.array(hist_type_one, dtype=float), np.array(hist, dtype=float)))

        else:

            still_data = False
            continue

    if hist is not None and hist_fractions is not None:
        x = np.linspace(bounds_ylo, bounds_yhi, args.num_bins)
        fig, ax1 = plt.subplots()
        ax1.plot(x, hist, color='b')
        ax1.set_xlabel('Distance from substrate base')
        ax1.set_ylabel('Normalized type 1 and 2 atom frequency', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(x, hist_fractions, color='darkred')
        ax2.set_ylabel('Type 1 atom fraction', color='darkred')
        ax2.tick_params('y', colors='darkred')

        plt.title(args.plot_title)

        plt.savefig(args.output)


def main():
    """
    Parse arguments and execute xyz file processing.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, dest='data', default=None, help='xyz file')
    parser.add_argument('-output', type=str, dest='output', default=None, help='File name for output files')
    parser.add_argument('-step', type=int, dest='step', default=0, help='Which step in the xyz trajectory to analyze')
    parser.add_argument('-num_atoms', type=int, dest='num_atoms', default=0,
                        help='Number (constant) of atoms in trajectory')
    parser.add_argument('-num_bins', type=int, dest='num_bins', default=10, help='Number of histogram bins')
    parser.add_argument('-plot_title', type=str, dest='plot_title', default='Density Profile', help='Plot title')

    args = parser.parse_args()
    process_xyz(args)


if __name__ == "__main__":
    main()
