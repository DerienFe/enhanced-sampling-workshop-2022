#Here we explore the NaCl FES energy landscape on Na-Cl distance. place random bias.
# the workflow:
# 1. simulate system 10000 steps, log the NaCl distance list.
# 2. generate partial Markov matrix.
# 3. 

## import required packages
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
from tqdm import tqdm
import mdtraj
from util import *
sys.path.append("..")
from openmm.app import *
from openmm.unit import *
from dham import DHAM


psf_file = 'toppar/step3_input.psf' # Path
pdb_file = 'toppar/step3_input_2A.pdb' # Path
psf = omm_app.CharmmPsfFile(psf_file)
pdb = omm_app.PDBFile(pdb_file)

with open("output_files/NaCl_solvated_system", 'r') as file_handle:
    xml = file_handle.read()
system = omm.XmlSerializer.deserialize(xml)
platform = omm.Platform.getPlatformByName('CUDA')
#### setup an OpenMM context

T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
steps = 100000  # total steps
propagation_step = 1000

integrator = omm.LangevinIntegrator(T*kelvin, #Desired Integrator
                                    fricCoef/picoseconds,
                                    stepsize*femtoseconds) 
## construct an OpenMM context
context = omm.Context(system, integrator)   # you may pass platform as a third positional argument


st = time.time()

s = time.time()
print("Setting up the simulation")

# Minimizing step
context.setPositions(pdb.positions)
state = context.getState(getEnergy = True)
energy = state.getPotentialEnergy()

for i in tqdm(range(50), desc="Minimization"):
    omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

print("Minimization done in", time.time() - s, "seconds")
s = time.time()

###############################################
# here we start the loop:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################
def DHAM_it(CV, ):
    d = DHAM()
    d.setup(CV, T, k_val, centers)

    d.lagtime = 1
    d.numbins = 0#?
    results = d.run(biased = True, plot=False)
    return

def propagate(context, steps=10000, dcdfreq=100, pdb=pdb, platform=platform, max_propagation=30, stepsize=stepsize):
    """
    propagate the system for a number of steps.
    """
    NaCl_dist = []
    for i in range(max_propagation):
        file_handle = open(f"trajectories/NaCl_exploring_traj_{i}.dcd", 'bw')
        dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)

        for j in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {i}"):
            integrator.step(dcdfreq) #advance dcd freq steps and stop to record.
            state = context.getState(getPositions=True)
            dcd_file.writeModel(state.getPositions(asNumpy=True))
        file_handle.close()

        #now we have the trajectory, we can calculate the Markov matrix.
        
        top = mdtraj.load_psf(psf_file)
        traj = mdtraj.load_dcd(f"trajectories/NaCl_exploring_traj_{i}.dcd", top=top)

        dist = mdtraj.compute_distances(traj, [[0, 1]]) *10 #unit in A #get distance over the traj (in this propagation)
        np.savetxt(f"trajectories/NaCl_exploring_traj_{i}.txt", dist) #save the distance to a txt file. 
        NaCl_dist.append(dist)           

propagate(context, steps=propagation_step, dcdfreq=dcdfreq, pdb=pdb, platform=platform, max_propagation=30, stepsize=stepsize)
