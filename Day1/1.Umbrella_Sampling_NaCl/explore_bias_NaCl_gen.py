#Here we explore the NaCl FES energy landscape on Na-Cl distance. place random bias.
# the workflow:
# propagate the system for a number of steps.
#    we have to start from loading the molecule, FF
#    then we put bias, log the bias form in txt file. (random guess around starting position if first propagation)
#    then we propagate the system for a number of steps.
#    use DHAM, feed in the NaCl distance, bias params for current propagation, get free energy landscape
#    use the partial free energy landscape to generate next bias params.
#    repeat.
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
platform = omm.Platform.getPlatformByName('CUDA')
psf_file = 'toppar/step3_input.psf' # Path
pdb_file = 'toppar/step3_input_2A.pdb' # Path
T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
propagation_step = 10000

def random_initial_bias(initial_position):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    rng = np.random.default_rng()
    a = np.ones(10)
    b = rng.uniform(initial_position-0.5, initial_position+0.5, 10) #min/max of preloaded NaCl fes x-axis.
    c = rng.uniform(1, 5.0, 10)
    return (a,b,c)
    
def DHAM_it(CV, gaussian_params, T=300, lagtime=1, numbins=60):
    """
    intput:
    CV: the collective variable we are interested in. Na-Cl distance.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM()
    d.setup(CV, T, gaussian_params)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=False)
    return results

def propagate(context, steps=10000, dcdfreq=100,  platform=platform, max_propagation=30, stepsize=stepsize):
    """
    propagate the system for a number of steps.
    we have to start from loading the molecule, FF
    then we define global params for bias (random guess around starting position if first propagation)
    
    each time, we generate new bias
    and propagate the system for a number of steps.
    use DHAM, feed in the NaCl distance, bias params for current propagation, get free energy landscape
    use the partial free energy landscape to generate next bias params.
    repeat.
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
        
        test_result = DHAM_it(dist, gaussian_params, T=300, lagtime=1, numbins=60)
        NaCl_dist.append(dist)           


###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":
    
    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)

    with open("output_files/NaCl_solvated_system", 'r') as file_handle:
        xml = file_handle.read()
    system = omm.XmlSerializer.deserialize(xml)
    platform = omm.Platform.getPlatformByName('CUDA')

    #### setup an OpenMM context
    
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
    gaussian_params = random_initial_bias(initial_position = 2.65)

    propagate(context, steps=propagation_step, dcdfreq=dcdfreq, platform=platform, max_propagation=30, stepsize=stepsize)
