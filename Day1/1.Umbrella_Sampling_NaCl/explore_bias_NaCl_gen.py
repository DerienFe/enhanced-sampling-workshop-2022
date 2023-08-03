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
stepsize = 2    # MD integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
propagation_step = 10000
max_propagation = 10
num_bins = 150 #for qspace used in DHAM and etc.

def random_initial_bias(initial_position):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    rng = np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(10) * 0.1
    b = rng.uniform(initial_position-0.5, initial_position+0.5, 10) #min/max of preloaded NaCl fes x-axis.
    c = rng.uniform(1, 5.0, 10)
    return np.concatenate((a,b,c), axis=None)
    
def DHAM_it(CV, gaussian_params, T=300, lagtime=2, numbins=num_bins):
    """
    intput:
    CV: the collective variable we are interested in. Na-Cl distance.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,b,c)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params)
    d.setup(CV, T)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True)
    return results

def propagate(context, gaussian_params, prop_index, NaCl_dist, steps=10000, dcdfreq=100,  platform=platform, stepsize=stepsize, num_bins=num_bins):
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
    
    
    file_handle = open(f"trajectories/explore_traj/NaCl_exploring_traj_{prop_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)

    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        integrator.step(dcdfreq) #advance dcd freq steps and stop to record.
        state = context.getState(getPositions=True)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #now we have the trajectory, we can calculate the Markov matrix.
    
    top = mdtraj.load_psf(psf_file)
    traj = mdtraj.load_dcd(f"trajectories/explore_traj/NaCl_exploring_traj_{prop_index}.dcd", top=top)

    dist = mdtraj.compute_distances(traj, [[0, 1]]) *10 #unit in A #get distance over the traj (in this propagation)
    np.savetxt(f"trajectories/NaCl_exploring_traj_{prop_index}.txt", dist) #save the distance to a txt file. 

    
    
    #we concatenate the new dist to the old dist.
    # NaCl_dist is a list of renewed dist.
    combined_dist = np.concatenate((NaCl_dist[-1], dist), axis=None)
    NaCl_dist.append(combined_dist)


    #plot it.
    plt.plot(combined_dist)
    plt.show()
    x, mU2, A, MM = DHAM_it(combined_dist.reshape(-1, 1), gaussian_params, T=300, lagtime=1, numbins=num_bins)
    
    cur_pos = combined_dist[-1] #the last position of the traj. not our cur_pos is the CV distance.
    
    return cur_pos, NaCl_dist, MM #return the CV traj and the MM.

def minimize(context):
    st = time.time()
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in tqdm(range(50), desc="Minimization"):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    print("Minimization done in", time.time() - s, "seconds")
    s = time.time()
    return context, energy

def add_bias(system, gaussian_params, num_gaussian=10):
    """
    gaussian params: np.array([a,b,c])
    system: the openmm system object.
    """
    a = gaussian_params[:num_gaussian]
    b = gaussian_params[num_gaussian:2*num_gaussian]
    c = gaussian_params[2*num_gaussian:]
    potential = ' + '.join(f'a{i}*exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(num_gaussian))
    print(potential)
    custom_bias = omm.CustomBondForce(potential)
    
    for i in range(num_gaussian):
        custom_bias.addGlobalParameter(f'a{i}', a[i])
        custom_bias.addGlobalParameter(f'b{i}', b[i])
        custom_bias.addGlobalParameter(f'c{i}', c[i])
        custom_bias.addBond(0, 1)

    system.addForce(custom_bias)
    return system

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(2.4, 9, 150+1)
    target_state = 7 #find the closest state to 7A.
    """
    working_states = qspace[working_indices] #the NaCl distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state


###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":
    
    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)

    """with open("output_files/NaCl_solvated_system", 'r') as file_handle:
        xml = file_handle.read()
    system = omm.XmlSerializer.deserialize(xml)
    
    """
    forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

    system = forcefield.createSystem(psf.topology,
                                     nonbondedCutoff=1.0*nanometers,
                                     constraints=omm_app.HBonds)
    platform = omm.Platform.getPlatformByName('CUDA')
    #### setup an OpenMM context
    integrator = omm.LangevinIntegrator(T*kelvin, #Desired Integrator
                                        fricCoef/picoseconds,
                                        stepsize*femtoseconds) 
    qspace = np.linspace(2.4, 9, num_bins+1) #hard coded for now.
    NaCl_dist = [[]] #initialise the NaCl distance list.
    for i in range(max_propagation):
        if i == 0:
            print("propagation number 0 STARTING.")
            gaussian_params = random_initial_bias(initial_position = 2.65)
            biased_system = add_bias(system, gaussian_params)

            ## construct an OpenMM context
            context = omm.Context(biased_system, integrator)   
            context, energy = minimize(context)         #minimize the system

            ## MD run "propagation"
            cur_pos, NaCl_dist, M= propagate(context, 
                                              gaussian_params=gaussian_params, 
                                              NaCl_dist = NaCl_dist,
                                              prop_index=i,
                                              steps=propagation_step, 
                                              dcdfreq=dcdfreq, 
                                              platform=platform, 
                                              stepsize=stepsize)
            
            
            #finding the closest element in MM to the end point. 7A in np.linspace(2.4, 9, 150+1)
            #trim the zero rows and columns markov matrix to avoid 0 rows.
            #!!!do everything in index space. !!!
            cur_pos_index = np.digitize(cur_pos, qspace) #the big index on full markov matrix.

            working_MM, working_indices = get_working_MM(M) #we call working_index the small index. its part of the full markov matrix.
            final_index = np.digitize(7, qspace) #get the big index of desired 7A NaCl distance.
            
            
            farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index in qspace.

        else:
            print(f"propagation number {i} STARTING.")
            #renew the gaussian params using returned MM.

            gaussian_params = try_and_optim_M(working_MM, 
                                              working_indices = working_indices,
                                              num_gaussian=10, 
                                              start_state=cur_pos_index, 
                                              end_state=farest_index,
                                              plot = True,
                                              )

            ## construct an OpenMM context
            #we use context.setParameters() to update the bias potential.
            for j in range(10):
                context.setParameter(f'a{j}', gaussian_params[j])
                context.setParameter(f'b{j}', gaussian_params[j+10])
                context.setParameter(f'c{j}', gaussian_params[j+20])
            
            #we plot the total bias.
            test_gaussian_params = []
            for j in range(10):
                test_gaussian_params.append(context.getParameter(f'a{j}'))
                test_gaussian_params.append(context.getParameter(f'b{j}'))
                test_gaussian_params.append(context.getParameter(f'c{j}'))
            
            test_total_bias = np.zeros_like(qspace)
            for n in range(len(test_gaussian_params)//3):
                test_total_bias += gaussian(qspace, test_gaussian_params[3*n], test_gaussian_params[3*n+1], test_gaussian_params[3*n+2])
            plt.plot(qspace, test_total_bias)
            plt.show()
                
            ## MD run "propagation"
            cur_pos, NaCl_dist, M= propagate(context, 
                                              gaussian_params=gaussian_params, 
                                              NaCl_dist = NaCl_dist,
                                              prop_index=i,
                                              steps=propagation_step, 
                                              dcdfreq=dcdfreq, 
                                              platform=platform, 
                                              stepsize=stepsize)
            
            cur_pos_index = np.digitize(cur_pos, qspace) #update cur_pos_index

            working_MM, working_indices = get_working_MM(M)
            farest_index = working_indices[np.argmin(np.abs(working_indices - final_index))] #get the closest to the final index
             

        if working_indices[-1] > final_index or working_indices[-1] == final_index:
            print(f"we have sampled the fes beyongd 7A, stop propagating at number {i}")
            break
        else:
            print("continue propagating.")
            continue


    
    
    
    
    
    
    
    
