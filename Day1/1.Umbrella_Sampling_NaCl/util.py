#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023

import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import minimize

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 

def create_K_1D(fes, N=200, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    
    #input:
    #fes: the free energy profile, a 1D array.

    K = np.zeros((N,N)) #, dtype=np.float64
    for i in range(N-1):
        K[i, i + 1] = np.exp((fes[i+1] - fes[i]) / 2 / kT)
        K[i + 1, i] = np.exp((fes[i] - fes[i+1]) / 2 / kT)
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K

def kemeny_constant_check(N, mfpt, peq):
    kemeny = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

#define a function calculating the mean first passage time
def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1)) #, dtype=np.float64
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N)) #, dtype=np.float64
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def bias_K_1D(K, total_bias, kT=0.5981):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  # Calculate K_biased
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))
    
    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

def compute_free_energy(K, kT=0.5981):
    """
    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]) #normalize the eigenvector
    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #print('sum of the peq is:', np.sum(peq))

    #calculate the free energy
    F = -kT * np.log(peq + 1e-6) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]


def try_and_optim(K, num_gaussian=10, start_state=0, end_state=0):
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params
    """
    best_mfpt = 1000000000000 #initialise the best mfpt np.inf
    for i in range(1000): 
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian)
        b = rng.uniform(2.41, 9, num_gaussian) #min/max of preloaded NaCl fes x-axis.
        c = rng.uniform(1, 5.0, num_gaussian) 
        
        total_bias = np.zeros(50) # we were using first 50 points of the fes
        for j in range(num_gaussian):
            total_bias += gaussian(np.arange(50), a[j], b[j], c[j])
        
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]
        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        print("random try:", i, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = (a, b, c)
            
    print("best mfpt:", best_mfpt)
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, K, kT=0.5981, start_state = start_state, end_state = end_state):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros(50)
        for j in range(num_gaussian):
            total_bias += gaussian(np.arange(50), a[j], b[j], c[j])
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]
        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        return mfpt_biased

    res = minimize(mfpt_helper, 
                   best_params, 
                   args=(K,), 
                   method='Nelder-Mead', 
                   bounds= [(0, 1)]*10 + [(2.41, 9)]*10 + [(1.0, 5.0)]*10, #add bounds to the parameters
                   tol=1e-1)

    return res.x
