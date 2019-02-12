# 2D PVD simulation that more closely follows Dan's paper than pvd_2D.py
import sys
sys.path.append('/home/swansonk1/DASH-7-9-2018/md_engine/build/python/build/lib.linux-x86_64-2.7')
import time
from DASH import *
import argparse
import copy
import numpy as np

def pvd_simulation(args):
    """
    Run a physical vapor-deposited glass simulation
    :param args: Simulation parameters
    """

    assert(args.num_simulations == len(args.substrate_temp))

    for sim_number in range(args.num_simulations):

        # Set initial DASH parameters
        start_time = time.time()
        state = State()
        state.deviceManager.setDevice(0)
        state.periodicInterval = 7
        state.shoutEvery = 50000
        state.rCut = 2.5
        state.padding = 0.5
        state.seedRNG()
        state.dt = 0.005

        # Establish some file names
        initial_xyz = "init_" + args.output + '_' + str(sim_number)
        intermediate_xyz = "intermediate_" + args.output + '_' + str(sim_number)
        restart_file = "restart_" + args.output + '_' + str(sim_number)

        # Set bounds and 2D attributes
        state.bounds = Bounds(state, lo=Vector(0, 0, 0), hi=Vector(args.x_len, 50, 0))
        state.is2d = True
        state.setPeriodic(2, False)
        state.setPeriodic(1, False)

        # Set atom types and interaction parameters
        state.atomParams.addSpecies(handle='substrate', mass=1)
        state.atomParams.addSpecies(handle='type1', mass=1)
        state.atomParams.addSpecies(handle='type2', mass=1)
        ljcut = FixLJCut(state, handle='ljcut')
        state.activateFix(ljcut)
        ljcut.setParameter(param='eps', handleA='type1', handleB='type1', val=1)
        ljcut.setParameter(param='sig', handleA='type1', handleB='type1', val=1)
        ljcut.setParameter(param='eps', handleA='type1', handleB='type2', val=1.5)
        ljcut.setParameter(param='sig', handleA='type1', handleB='type2', val=0.8)
        ljcut.setParameter(param='eps', handleA='type2', handleB='type2', val=0.5)
        ljcut.setParameter(param='sig', handleA='type2', handleB='type2', val=0.88)
        ljcut.setParameter(param='eps', handleA='type1', handleB='substrate', val=1.0)
        ljcut.setParameter(param='sig', handleA='type1', handleB='substrate', val=0.75)
        ljcut.setParameter(param='eps', handleA='type2', handleB='substrate', val=1.0)
        ljcut.setParameter(param='sig', handleA='type2', handleB='substrate', val=0.75)
        ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=0.1)
        ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=0.76)
        substrate_init_bounds = Bounds(state, lo=Vector(state.bounds.lo[0], state.bounds.lo[1], 0),
                                       hi=Vector(state.bounds.hi[0], 3.0, 0))

        # Set 2D fix
        fix2d = Fix2d(state, handle='fix2d', applyEvery=1)
        state.activateFix(fix2d)

        # Initialize substrate atom positions
        print('going to populate')
        InitializeAtoms.populateRand(state, bounds=substrate_init_bounds,
                                     handle='substrate', n=args.num_substrate_atoms, distMin=0.6)

        print('populated')

        # Initialize substrate atom velocities
        InitializeAtoms.initTemp(state, 'all', args.substrate_temp[sim_number])

        # Create substrate atom group
        state.createGroup('sub')
        state.addToGroup('sub', [a.id for a in state.atoms])

        # Define spring functions and activate spring tethering
        def spring_func(atom):
            atom.pos[1] = (substrate_init_bounds.lo[1] + substrate_init_bounds.hi[1]) / 2
            return atom.pos

        def spring_func_equiled(atom):
            return atom.pos
        fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub', k=5, tetherFunc=spring_func,
                                    multiplier=Vector(0.2, 1, 1))
        state.activateFix(fixSpring)

        # Create Nose Hoover thermostat
        print('Creating Nose Hoover thermostat')
        fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub')
        fixNVT.setTemperature(temp=args.substrate_temp[sim_number], timeConstant=100*state.dt)
        print('Activating Nose Hoover thermostat')
        state.activateFix(fixNVT)

        # Create integrators
        integrator = IntegratorVerlet(state)
        integratorRelax = IntegratorGradientDescent(state)

        # Create initial trajectory writer
        writerxyz = WriteConfig(state, handle='writer', fn=initial_xyz, format='xyz', writeEvery=10000)
        state.activateWriteConfig(writerxyz)

        # Minimize substrate configuration
        integratorRelax.run(950000, 1)
        print('FINISHED FIRST RUN')

        # Re-tether atoms with new spring constant
        fixSpring.tetherFunc = spring_func_equiled
        fixSpring.updateTethers()
        fixSpring.k = 1000
        fixSpring.multiplier = Vector(1, 1, 1)

        # Create a harmonic wall
        wallDist = 10.0
        topWall = FixWallHarmonic(state, handle='wall', groupHandle='all', origin=Vector(0, 13, 0),
                                  forceDir=Vector(0, -1, 0), dist=wallDist, k=args.wall_spring_const)
        state.activateFix(topWall)

        # Integrate with the wall present
        integrator.run(50000)
        print('FINISHED SECOND RUN')
        state.deactivateWriteConfig(writerxyz)

        # Initialize deposition parameters, atom groups, and files
        newVaporGroup = 'vapor'
        vaporTemp = 1.0
        state.createGroup('film')
        state.createGroup('bulk')
        num_film_atoms = 0
        num_bulk_atoms = 0
        f = open(args.PE_file + '_' + str(sim_number) + '.txt', 'w')
        f.write('# ' + 'Deposition Step' + '    ' + 'Bulk Potential Energy' + '\n')
        f.close()

        # Deposit atoms
        for i in range(args.deposition_runs):
            print('Deposition step {}'.format(i))

            # Set new y bound
            maxY = max(a.pos[1] for a in state.atoms)
            print('max Y', maxY)
            newTop = maxY + wallDist
            hi = state.bounds.hi
            hi[1] = newTop
            state.bounds.hi = hi
            topWall.origin = Vector(0, state.bounds.hi[1], 0)
            print('Wall y-pos: %f' % newTop)

            # Deposit new set of atoms
            populateBounds = Bounds(state, lo=Vector(state.bounds.lo[0], maxY + 3, 0),
                                    hi=Vector(state.bounds.hi[0], maxY + 5, 0))
            num_type1 = np.random.randint(6, 8)
            num_type2 = np.random.randint(3, 5)
            InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type1', n=num_type1, distMin=1)
            InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type2', n=num_type2, distMin=1)

            # Add new atoms to groups and keep track of total count
            newAtoms = []
            for k in range(1, 1+num_type1+num_type2):
                na = state.atoms[-k]
                newAtoms.append(state.atoms[-k])
                print('New atom: {}, pos ({},{},{})'.format(na.id, na.pos[0], na.pos[1], na.pos[2]))
            state.createGroup(newVaporGroup)
            state.addToGroup(newVaporGroup, [a.id for a in newAtoms])
            state.addToGroup('film', [a.id for a in newAtoms])
            num_film_atoms += len(newAtoms)

            # Initialize new atom temperatures and delete temporary group
            InitializeAtoms.initTemp(state, newVaporGroup, vaporTemp)
            state.deleteGroup(newVaporGroup)

            # Run integrator for all but 100 deposition steps
            integrator.run(args.num_turns_deposition[sim_number]-100)

            # Set up writers and recorders to monitor trajectory, restart info, and bulk energy
            thermalImageName = intermediate_xyz + '_' + str(i)
            writer = WriteConfig(state, handle='writer', fn=thermalImageName, format='xyz', writeEvery=1)
            state.activateWriteConfig(writer)
            restartFileName = restart_file + '_' + str(i)
            restart = WriteConfig(state, handle='restart', fn=restartFileName, format='xml', writeEvery=1)
            integrator.run(99)
            state.activateWriteConfig(restart)
            new_bulk_atoms = [a.id for a in newAtoms if args.bulk_lo <= a.pos[1] <= args.bulk_hi]
            if len(new_bulk_atoms) > 0:
                state.addToGroup('bulk', new_bulk_atoms)
                num_bulk_atoms += len(new_bulk_atoms)
            if num_bulk_atoms > 0:
                engDataSimple = state.dataManager.recordEnergy(handle='bulk', mode='scalar', interval=1, fixes=[ljcut])
            integrator.run(1)
            if num_bulk_atoms > 0:
                for energies in engDataSimple.vals:
                    f = open(args.PE_file + '_' + str(sim_number) + '.txt', 'a')
                    f.write(str(i) + '    ' + str(energies / float(num_bulk_atoms)) + '\n')
                    f.close()
                state.dataManager.stopRecord(engDataSimple)
            state.deactivateWriteConfig(restart)
            state.deactivateWriteConfig(writer)

            # Print current simulation temperature and elapsed time
            curTemp = sum([a.vel.lenSqr()/3.0 for a in state.atoms]) / len(state.atoms)
            print('cur temp %f ' % curTemp)
            currentTime = time.time()
            elapsedTime = currentTime - start_time
            print(elapsedTime)

        # Record final trajectory positions of completed film
        writer = WriteConfig(state, handle='writer', fn='final_' + args.output + '_' + str(sim_number), format='xyz',
                             writeEvery=1)
        state.activateWriteConfig(writer)
        integrator.run(1)
        state.deactivateWriteConfig(writer)

        # Compute inherent structure energy of bulk atoms
        integratorRelax.run(args.num_turns_final_relaxation - 1, 1)
        engDataSimple = state.dataManager.recordEnergy(handle='bulk', mode='scalar', interval=1, fixes=[ljcut])
        integratorRelax.run(1, 1)
        f = open(args.EIS_file + '_' + str(sim_number) + '.txt', 'w')
        f.write('# ' + 'Bulk Inherent Structure Energy' + '\n')
        for energies in engDataSimple.vals:
                f.write(str(energies / float(num_bulk_atoms)))
        f.close()


def main():
    """
    Parse arguments and execute PVD film simulation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, dest='output', default='pvd_2D', help='File name suffix for output files')
    parser.add_argument('--num_substrate_atoms', type=int, dest='num_substrate_atoms', default=104,
                        help='Number of substrate atoms')
    parser.add_argument('--x_len', type=float, dest='x_len', default=30.0, help='Box length in x dimension')
    parser.add_argument('--wall_spring_const', type=int, dest='wall_spring_const', default=5,
                        help='Spring constant for harmonic wall')
    parser.add_argument('--num_turns_deposition', type=int, dest='num_turns_deposition', nargs='*', default=[1000000],
                        help='Number of integrator turns for each deposition step')
    parser.add_argument('--deposition_runs', type=int, dest='deposition_runs', default=55,
                        help='Number of depositions')
    parser.add_argument('--substrate_temp', type=float, dest='substrate_temp', nargs='*', default=[0.2345],
                        help='Substrate temperature in LJ units')
    parser.add_argument('--bulk_lo', type=float, dest='bulk_lo', default=10.0,
                        help='Lower y-dimension bound on the bulk film region')
    parser.add_argument('--bulk_hi', type=float, dest='bulk_hi', default=25.0,
                        help='Upper y-dimension bound on the bulk film region')
    parser.add_argument('--num_turns_final_relaxation', type=int, dest='num_turns_final_relaxation', default=10000000,
                        help='Number of integrator turns for each deposition step')
    parser.add_argument('--EIS_file', type=str, dest='EIS_file', default='EIS',
                        help='Text file containing final value of bulk inherent structural energy')
    parser.add_argument('--PE_file', type=str, dest='PE_file', default='PE',
                        help='Text file containing successive values of bulk potential energy')
    parser.add_argument('--num_simulations', type=int, dest='num_simulations', default=1,
                        help='Number of simulations to run')

    args = parser.parse_args()

    pvd_simulation(args)


if __name__ == "__main__":
    main()
