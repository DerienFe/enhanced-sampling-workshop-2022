## import required packages
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
from tqdm.notebook import tqdm_notebook
import mdtraj
from util import *
sys.path.append("..")

psf_file = 'toppar/step3_input.psf' # Path
pdb_file = 'toppar/step3_input.pdb' # Path
psf = omm_app.CharmmPsfFile(psf_file)
pdb = omm_app.PDBFile(pdb_file)

#params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #old way.
## Create an OpenMM system
##system = psf.createSystem(params) #old way.
from openmm.app import *
from openmm.unit import *
forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')

system = forcefield.createSystem(psf.topology,
                                 nonbondedCutoff=1.0*nanometers, 
                                 constraints=omm_app.HBonds)


#have a little play with pre-loaded unbiased data.
# try get the K matrix and then the mfpt from start_state = 5.9A to end_state = 2.65A
unb_bins, unb_profile = np.load("Unbiased_Profile.npy")

#find the closest bin to 5.9A and 2.65A: 26 2 
#state for 2.6A and 15A is 26 and 95
start_state = np.argmin(np.abs(unb_bins - 5.9))
end_state = np.argmin(np.abs(unb_bins - 15))

#we truncate, take first 1/4
unb_bins = unb_bins#[:len(unb_bins)//4]
unb_profile = unb_profile#[:len(unb_profile)//4]
print(unb_bins.shape, unb_profile.shape)


N = 200 #discretized fes.
K = create_K_1D(unb_profile, N=N, kT=0.5981)
[peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(K)
mfpts = mfpt_calc(peq, K)
mfpt = mfpts[26,95]
kemeny_constant_check(N, mfpts, peq)



#plot the reconstructed F
#2.418102979660034 8.924837112426758 min/max of preloaded NaCl fes x-axis.
#2.41 28.8 A for whole FES.
#the 3rd element of unb_profile is zero. 
plt.plot(unb_bins, F - F.min(), label="reconstructed F")
plt.plot(unb_bins, unb_profile - unb_profile.min(), label="unbiased F")
plt.legend()
plt.xlabel("Na-Cl distance (A)")
plt.show()

#print(start_state, end_state)

#now we can try mfpt optimized bias to push the system from 5.9A to 2.65A
# note bias potential is 10 gaussianfunctions added together.
# then we random try 1000 times and local optmize the gaussian params.

gaussian_params = try_and_optim(K, num_gaussian=10)

#print(gaussian_params)
#unpack params
num_gaussian = 10
a = gaussian_params[:num_gaussian]
b = gaussian_params[num_gaussian:2*num_gaussian]
c = gaussian_params[2*num_gaussian:]
#apply it and plot the new F
total_bias = np.zeros(200)
for j in range(10):
    total_bias += gaussian(np.arange(200), a[j], b[j], c[j])
K_biased = bias_K_1D(K, total_bias, kT=0.5981)
[peq_biased, F_biased, evectors, evalues, evalues_sorted, index] = compute_free_energy(K_biased)

mfpts_biased = mfpt_calc(peq_biased, K_biased)
mfpt_biased = mfpts_biased[start_state, end_state]

kemeny_constant_check(N, mfpts_biased, peq_biased)

#plot the reconstructed F
plt.plot(unb_bins, F_biased - F_biased[26], label="biased F")
plt.plot(unb_bins, unb_profile - unb_profile[26], label="unbiased F")
plt.title(f"mfpt biased after try and optim. mfpt: {mfpt:.2f} mfpt_biased: {mfpt_biased:.2f}")
plt.legend()
plt.show()


#now that we have params to bias the system
# we plug in the openmm system and run the simulation.
potential = ' + '.join(f'a{i}*exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(num_gaussian))
print(potential)
custom_bias = omm.CustomBondForce(potential)
for i in range(num_gaussian):
    custom_bias.addGlobalParameter(f'a{i}', a[i])
    custom_bias.addGlobalParameter(f'b{i}', b[i])
    custom_bias.addGlobalParameter(f'c{i}', c[i])

custom_bias.addBond(0, 1)
system.addForce(custom_bias)


## save the OpenMM system
with open("output_files/NaCl_solvated_system_mfpt_bias_3", 'w') as file_handle:
    file_handle.write(omm.XmlSerializer.serialize(system))

print("All Done")
