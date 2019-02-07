import sys
sys.path.append('/home/swansonk1/DASH-7-9-2018/md_engine/build/python/build/lib.linux-x86_64-2.7')
import time
from DASH import *
from math import *
import argparse
import copy

def pvd_simulation(args):
    """
    Run a physical vapor-deposited glass simulation
    :param args: Simulation parameters
    :return:
    """

    start_time = time.time()
    state = State()
    state.deviceManager.setDevice(0)
    state.periodicInterval = 7
    state.shoutEvery = 50000
    state.rCut = 2.5
    state.padding = 0.5
    state.seedRNG()

    initial_xyz = "init_" + args.output
    intermediate_xyz = "intermediate_" + args.output
    restart_file = "restart_" + args.output
    args.num_turns_deposition = 1000000
    ################################################
    # Substrate - initial layer parameters
    ################################################
    nAtoms = 330
    nLayers = 5

    L = float(nAtoms)/float(nLayers)
    print "sidelengths of ", L
    args.substrate_temp = 0.2345

    state.bounds = Bounds(state, lo=Vector(0,0,0), hi=Vector(L,50,0))
    state.is2d=True
    state.setPeriodic(2, False)
    state.setPeriodic(1, False)

    state.atomParams.addSpecies(handle='substrate',mass=1)
    state.atomParams.addSpecies(handle='type1',mass=1)
    state.atomParams.addSpecies(handle='type2',mass=1)
    ljcut = FixLJCut(state, handle='ljcut')
    state.activateFix(ljcut)
    ljcut.setParameter(param='eps', handleA='type1', handleB='type1', val=1)
    ljcut.setParameter(param='sig', handleA='type1', handleB='type1', val=1)
    ljcut.setParameter(param='eps', handleA='type1', handleB='type2', val=1.5)
    ljcut.setParameter(param='sig', handleA='type1', handleB='type2', val=0.8)
    ljcut.setParameter(param='rCut', handleA='type1', handleB='type2', val=2.0)
    ljcut.setParameter(param='eps', handleA='type2', handleB='type2', val=0.5)
    ljcut.setParameter(param='sig', handleA='type2', handleB='type2', val=0.88)
    ljcut.setParameter(param='rCut', handleA='type2', handleB='type2', val=2.2)
    ljcut.setParameter(param='eps', handleA='type1', handleB='substrate', val=1.0)
    ljcut.setParameter(param='sig', handleA='type1', handleB='substrate', val=0.75)
    ljcut.setParameter(param='eps', handleA='type2', handleB='substrate', val=1.0)
    ljcut.setParameter(param='sig', handleA='type2', handleB='substrate', val=0.7)
    ljcut.setParameter(param='eps', handleA='substrate', handleB='substrate', val=0.1)
    ljcut.setParameter(param='sig', handleA='substrate', handleB='substrate', val=0.76)
    substrate_init_bounds = Bounds(state, lo=Vector(state.bounds.lo[0], state.bounds.lo[1], 0),
                                        hi=Vector(state.bounds.hi[0], 14.5, 0))

    fix2d = Fix2d(state, handle='fix2d', applyEvery=1)
    state.activateFix(fix2d)
    print 'going to populate'
    InitializeAtoms.populateRand(state, bounds=substrate_init_bounds,
                                  handle='substrate', n= nAtoms, distMin = 0.6)

    print 'populated'

    InitializeAtoms.initTemp(state, 'all', args.substrate_temp)
    state.createGroup('sub')
    state.addToGroup('sub', [a.id for a in state.atoms])
    def spring_func(atom):
        atom.pos[1] = (substrate_init_bounds.lo[1] + substrate_init_bounds.hi[1]) / 2
        return atom.pos
    def spring_func_equiled(atom):
        return atom.pos
    fixSpring = FixSpringStatic(state, handle='substrateSpring', groupHandle='sub',
                            k=10, tetherFunc=spring_func,
                            multiplier=Vector(0.2, 1, 1))
    state.activateFix(fixSpring)
    state.dt = 0.005
    print('Creating Nose Hoover thermostat')
    fixNVT = FixNoseHoover(state, handle='nvt', groupHandle='sub')
    fixNVT.setTemperature(temp=args.substrate_temp, timeConstant=100*state.dt)
    print('Activating Nose Hoover thermostat')
    state.activateFix(fixNVT)
    integrator = IntegratorVerlet(state)
    integratorRelax = IntegratorGradientDescent(state)
    writerxyz = WriteConfig(state, handle='writer', fn=initial_xyz, format='xyz', writeEvery=10000)
    state.activateWriteConfig(writerxyz)
    integratorRelax.run(500000, 1)
    fixSpring.k = 1.0
    integratorRelax.run(450000, 1)
    print 'FINISHED FIRST RUN'
    state.dt = 0.005
    InitializeAtoms.initTemp(state, 'all', args.substrate_temp)
    fixSpring.tetherFunc = spring_func_equiled
    fixSpring.updateTethers() # tethering to the positions they fell into
    fixSpring.k = 1000
    fixSpring.multiplier = Vector(1, 1, 1) # now spring holds in both dimensions
    InitializeAtoms.initTemp(state, 'all', args.substrate_temp)
    wallDist = 10
    topWall = FixWallHarmonic(state, handle='wall', groupHandle='all',
                          origin=Vector(0, state.bounds.hi[1], 0),
                          forceDir=Vector(0, -1, 0), dist=wallDist, k=15)
    state.activateFix(topWall)
    integrator.run(50000)
    print 'FINISHED SECOND RUN'
    # Start deposition
    stoichiometric = [65, 35]
    newVaporGroup = 'vapor'
    vaporTemp = 0.5
    toDepIdx = 0
    state.createGroup('film')
    state.createGroup('bulk')
    num_film_atoms = 0
    num_bulk_atoms = 0
    state.deactivateWriteConfig(writerxyz)
    atomTypesToAdd = ''
    f = open(args.PE_file + '.txt', 'w')
    f.write('# ' + 'Deposition Step' + '    ' + 'Bulk Potential Energy' + '\n')
    f.close()
    for i in range(args.deposition_runs):
        print 'Deposition step {}'.format(i)
        toDeposit = copy.deepcopy(stoichiometric)
        if (atomTypesToAdd):
            type1ToAdd = 0
            type2ToAdd = 0
            for qq in range(len(atomTypesToAdd)):
                if (atomTypesToAdd[qq] == 'type1'):
                    type1ToAdd += 1
                elif (atomTypesToAdd[qq] == 'type2'):
                    type2ToAdd += 1
            toDeposit[0] += type1ToAdd
            toDeposit[1] += type2ToAdd
        maxY = max(a.pos[1] for a in state.atoms)
        print 'toDeposit: {}'.format(toDeposit)
        print 'max Y', maxY
        newTop = max(maxY + 20 + wallDist, state.bounds.hi[1])
        hi = state.bounds.hi
        hi[1] = newTop
        state.bounds.hi = hi
        topWall.origin = Vector(0, state.bounds.hi[1], 0)
        print('Wall y-pos: %f' % newTop)
        populateBounds = Bounds(state,
                            lo=Vector(state.bounds.lo[0], newTop-wallDist-12, 0),
                            hi=Vector(state.bounds.hi[0], newTop-wallDist-8, 0))
        InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type1',
                                 n=toDeposit[0], distMin=1)
        InitializeAtoms.populateRand(state, bounds=populateBounds, handle='type2',
                                 n=toDeposit[1], distMin=1)
        newAtoms = []
        for k in range(1, 1+sum(toDeposit)):
            na = state.atoms[-k]
            newAtoms.append(state.atoms[-k])
            print('New atom: {}, pos ({},{},{})'.format(na.id, na.pos[0], na.pos[1], na.pos[2]))
        state.createGroup(newVaporGroup)
        state.addToGroup(newVaporGroup, [a.id for a in newAtoms])
        state.addToGroup('film', [a.id for a in newAtoms])
        num_film_atoms += len(newAtoms)
        InitializeAtoms.initTemp(state, newVaporGroup, vaporTemp)
        state.deleteGroup(newVaporGroup)
        for a in newAtoms:
            a.vel[1] = min(-abs(a.vel[1]), -0.2)
        curTemp = sum([a.vel.lenSqr()/3.0 for a in newAtoms]) / len(newAtoms)
        for a in newAtoms:
            a.vel *= sqrt(vaporTemp / curTemp)
        integrator.run(args.num_turns_deposition-100)

        toDepIdx += 1
        toDepIdx = toDepIdx % len(toDeposit)
        thermalImageName = intermediate_xyz + '_' + str(i)
        writer = WriteConfig(state,handle='writer',fn=thermalImageName, format='xyz',writeEvery=1)
        state.activateWriteConfig(writer)
        restartFileName =  restart_file + '_' + str(i)
        restart = WriteConfig(state,handle='restart',fn=restartFileName,format='xml',writeEvery=1)
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
                f = open(args.PE_file + '.txt', 'a')
                f.write(str(i) + '    ' + str(energies / float(num_bulk_atoms)) + '\n')
                f.close()
            state.dataManager.stopRecord(engDataSimple)
        state.deactivateWriteConfig(restart)
        state.deactivateWriteConfig(writer)
        curTemp = sum([a.vel.lenSqr()/3.0 for a in state.atoms]) / len(state.atoms)
        print 'cur temp %f ' % curTemp
        currentTime = time.time()
        elapsedTime = currentTime - start_time
        print(elapsedTime)

    # Compute final integrate
    writer = WriteConfig(state,handle='writer',fn='final_' + args.output, format='xyz',writeEvery=1)
    state.activateWriteConfig(writer)
    integrator.run(1)
    state.deactivateWriteConfig(writer)

    # Compute inherent structure energy
    integratorRelax.run(args.num_turns_final_relaxation - 1, 1)
    engDataSimple = state.dataManager.recordEnergy(handle='bulk', mode='scalar', interval=1, fixes=[ljcut])
    integratorRelax.run(1, 1)
    f = open(args.EIS_file + '.txt', 'w')
    f.write('# ' + 'Bulk Inherent Structure Energy' + '\n')
    for energies in engDataSimple.vals:
            f.write(str(energies / float(num_bulk_atoms)))
    f.close()

def main():
    """
    Parse arguments and execute PVD film simulation.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-output', type=str, dest='output', default='pvd_2D', help='File name suffix for output files')
    parser.add_argument('-num_turns_deposition', type=int, dest='num_turns_deposition', default=1000000,
                        help='Number of integrator turns for each deposition step')
    parser.add_argument('-deposition_runs', type=int, dest='deposition_runs', default=55,
                        help='Number of depositions')
    parser.add_argument('-substrate_temp', type=float, dest='substrate_temp', default=0.2345,
                        help='Substrate temperature in LJ units')
    parser.add_argument('-bulk_lo', type=float, dest='bulk_lo', default=25.0,
                        help='Lower y-dimension bound on the bulk film region')
    parser.add_argument('-bulk_hi', type=float, dest='bulk_hi', default=65.0,
                        help='Upper y-dimension bound on the bulk film region')
    parser.add_argument('-num_turns_final_relaxation', type=int, dest='num_turns_final_relaxation', default=1000000,
                        help='Number of integrator turns for each deposition step')
    parser.add_argument('-EIS_file', type=str, dest='EIS_file', default='EIS',
                        help='Text file containing final value of bulk inherent structural energy')
    parser.add_argument('-PE_file', type=str, dest='PE_file', default='PE',
                        help='Text file containing successive values of bulk potential energy')

    args = parser.parse_args()

    pvd_simulation(args)


if __name__ == "__main__":
    main()
