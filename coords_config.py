import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import PIL
import os
from typing import List, Union


def round_up_to_even(f):
    """
    Round up to the nearest even number
    :param f: Input number
    :return: Nearest even number greater than f
    """
    return np.ceil(f / 2.) * 2


def round_down_to_even(f):
    """
    Round down to the nearest even number
    :param f: Input number
    :return: Nearest even number less than f
    """
    return np.floor(f / 2.) * 2


class Molecule:
    def __init__(self):
        """
        Class to hold information about one timestep of a simulation trajectory file, including atom positions,
        atom types, number of atoms, and timestep
        """
        self.x = []
        self.atom_types = []
        self.natoms = None
        self.timestep = None

    def read_dump_timestep(self, fnme: str, targettimestep: int):
        """
        Read atom information from the specified timestep
        :param fnme: File name of trajectory file
        :param targettimestep: Target timestep to collect atom information from
        """
        f = open(fnme)
        reading = True
        found_flag = False

        # Read file lines
        while reading:
            line = f.readline()
            # Check timestep
            if "TIMESTEP" in line:
                line = f.readline().split()
                self.timestep = int(line[0])

            # Check number of atoms
            elif "NUMBER OF ATOMS" in line:
                line = f.readline().split()
                n = int(line[0])
                self.natoms = n
                self.x = np.zeros((n, 3))
                self.atom_types = np.zeros(n)

            # Save atom types nad coordinates at the target timestep
            elif "ITEM: ATOMS" in line and self.timestep == targettimestep:
                if self.natoms == 0:
                    print("Error! No atoms!")
                    exit(1)
                for j in range(self.natoms):
                    line = f.readline().split()
                    idx = int(line[0])-1
                    self.atom_types[idx] = int(line[1])
                    self.x[idx][0] = float(line[2])
                    self.x[idx][1] = float(line[3])
                    self.x[idx][2] = float(line[4])

                reading = False
                found_flag = True

        f.close()

        if not found_flag:
            print("Error! Timestep = %d not found in %s" % (targettimestep, fnme))
            exit(1)

    def coord_config(self, outputfile: str, dimensions: List[Union[float, None]]):
        """
        Print atom types and coordinates to a .txt file
        :param outputfile: Output file name
        :param dimensions: Dimensions to capture in the output file, given as [xlo, xhi, ylo, yhi]
        """
        xlo = dimensions[0]
        xhi = dimensions[1]
        ylo = dimensions[2]
        yhi = dimensions[3]

        if xlo is None or xhi is None or ylo is None or yhi is None:
            check_dimensions = False
        else:
            check_dimensions = True

        out = open(outputfile+'.txt', 'w')

        for i in range(self.natoms):
            if check_dimensions:
                if xlo <= self.x[i][0] <= xhi and ylo <= self.x[i][1] <= yhi:
                    out.write(str(self.atom_types[i]) + '   ' + str(self.x[i][0]) + '   ' + str(self.x[i][1]))
                    out.write('\n')
            else:
                out.write(str(self.atom_types[i]) + '   ' + str(self.x[i][0]) + '   ' + str(self.x[i][1]))
                out.write('\n')

    def plot_config(self, outputfile: str, figsize: int = 250, trimfrac: float = 0.1, color1: str = 'orange',
                    color2: str = 'blue'):
        """
        Create images of atomic configurations saved as a .png file
        :param outputfile: Output file name
        :param figsize: Image size in pixels
        :param trimfrac: Trim fraction, which is used to expand image before cropping, a process intended to remove
        any extra white space or frame from matplotlib plot
        :param color1: Color of type 1 atoms
        :param color2: Color of type 2 atoms
        """

        dpi = 100  # dpi of images

        figsizeextra = round_up_to_even(figsize * (1 + trimfrac))
        figsizeextra_inch = figsizeextra / dpi

        fig = plt.figure(figsize=(figsizeextra_inch, figsizeextra_inch), dpi=dpi, frameon=False)

        axisrange = np.array([0, 0, 1, 1])
        fig.add_axes(axisrange)

        sel = self.atom_types == 1
        x, y = self.x[sel, 0], self.x[sel, 1]
        plt.scatter(x, y, s=1, color=color1)
        sel = self.atom_types == 2
        x, y = self.x[sel, 0], self.x[sel, 1]
        plt.scatter(x, y, s=1, color=color2)

        plt.axis('equal')

        plt.savefig('tmp.png')

        img = PIL.Image.open("tmp.png")
        os.remove("tmp.png")

        center = figsizeextra / 2.0
        lo = int(center - 0.5 * figsize)
        hi = int(center + 0.5 * figsize)
        area = (lo, lo, hi, hi)
        cropped_img = img.crop(area)
        cropped_img.save(outputfile+'.png')


def main(trajfile: str, timestep: int, outputfile: str, file_format: str, dimensions: List[Union[float, None]]):
    """
    Execute reading and analysis of a trajectory file
    :param trajfile: Trajectory file containing atom types and atom positions
    :param timestep: Simulation timestep to lift coordinates from
    :param outputfile: Filename for output coordinate file
    :param file_format: Format is set to coords or image
    :param dimensions: Dimensions to capture in the output file, given as [xlo, xhi, ylo, yhi]'
    """
    mol = Molecule()
    mol.read_dump_timestep(trajfile, timestep)

    if file_format == 'coords':
        mol.coord_config(outputfile, dimensions)
    elif file_format == 'image':
        mol.plot_config(outputfile, trimfrac=0.1)

