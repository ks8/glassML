import os
import argparse
import coords_config


def process_coordinates(args):
    """
    Process coordinate file
    :param args: Arguments passed from main()
    """
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    timestep = args.timestep
    dimensions = [args.xlo, args.xhi, args.ylo, args.yhi]
    file_format = args.file_format
    trimfrac = args.trimfrac
    figsize = args.figsize

    for i in range(args.traj_file_start, args.traj_file_start + args.num_trajectories):
        print(i)
        trajfile = args.data+'/'+str(i)+'/traj.atom'
        outputfile = args.output+'/'+args.type+'.'+'{:05d}'.format(i)

        coords_config.main(trajfile, timestep, outputfile, file_format, dimensions, trimfrac, figsize)


def main():
    """
    Parse arguments and execute coordinate processing. If dimensions are going to be specified, then all four of
    xlo, xhi, ylo, and yhi have to be specified.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, dest='data', default=None, help='Directory containing trajectory files')
    parser.add_argument('-output', type=str, dest='output', default=None, help='Directory name for output files')
    parser.add_argument('-file_format', type=str, dest='file_format', default=None,
                        help='Set format to coords or image')
    parser.add_argument('-timestep', type=int, dest='timestep', default=None, help='Simulation timestep to read')
    parser.add_argument('-xlo', type=float, dest='xlo', default=None, help='xlo dimension for reading coordinates')
    parser.add_argument('-xhi', type=float, dest='xhi', default=None, help='xhi dimension for reading coordinates')
    parser.add_argument('-ylo', type=float, dest='ylo', default=None, help='ylo dimension for reading coordinates')
    parser.add_argument('-yhi', type=float, dest='yhi', default=None, help='yhi dimension for reading coordinates')
    parser.add_argument('-type', type=str, dest='type', default=None, help='Specify liquid or glass')
    parser.add_argument('-traj_file_start', type=int, dest='traj_file_start', default=0,
                        help='Initial directory name for trajectory files: usually begins with 0 or 1')
    parser.add_argument('-num_trajectories', type=int, dest='num_trajectories', default=None,
                        help='Number of trajectory files to read')
    parser.add_argument('-trimfrac', type=float, dest='trimfrac', default=0.1,
                        help='Trim fraction for plotting')
    parser.add_argument('-figsize', type=float, dest='figsize', default=250,
                        help='Image size in pixels (square image)')

    args = parser.parse_args()

    process_coordinates(args)


if __name__ == "__main__":
    main()




