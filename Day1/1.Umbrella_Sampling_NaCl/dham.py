#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize
from util import gaussian


def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


def count_transitions(b, numbins, lagtime, endpt=None):
    """
    note the b is a 2D array, 
     row represents the trajectory, column represents the time.
    """
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime] - 1, endpt[k, i] - 1] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    #sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    return sumtr, trvec


class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    numbins = 150
    lagtime = 1

    def __init__(self, gaussian_params):
        #unpack it to self.a, self.b, self.c
        num_gaussian = len(gaussian_params)//3
        self.a = gaussian_params[:num_gaussian]
        self.b = gaussian_params[num_gaussian:2*num_gaussian]
        self.c = gaussian_params[2*num_gaussian:]
        return

    def setup(self, CV, T):
        self.data = CV
        self.KbT = 0.001987204259 * T
        return

    def build_MM(self, sumtr, trvec, biased=False):
        MM = np.empty(shape=sumtr.shape, dtype=np.longdouble)
        if biased:
            MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
            #qsp = self.qspace[1] - self.qspace[0] #step size between bins
            for i in range(sumtr.shape[0]):
                for j in range(sumtr.shape[1]):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        for k in range(trvec.shape[0]):
                            #u = 0.5 * self.k_val[k] * np.square(self.constr_val[k] - self.qspace - qsp / 2) / self.KbT #change this line to adapt our 10-gaussian bias.
                            #here we use the 10-gaussian bias. a,b,c are given.
                            u = np.zeros_like(self.qspace)
                            for n in range(len(self.a)):
                                u += gaussian(self.qspace, self.a[n], self.b[n], self.c[n])
                            #u = u - qsp/2 #adjust the bias so it is at the bin center.
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / 2)
                        if sump1 > 0:
                            MM[i, j] = sumtr[i, j] / sump1
                        else:
                            MM[i, j] = 0
            epsilon_offset = 1e-15
            MM = MM / (np.sum(MM, axis=1)[:, None]+epsilon_offset) #normalize the M matrix #this is returning NaN?.
            
        else:
            MM[:, :] = sumtr / np.sum(sumtr, axis=1)[:, None]
        return MM

    def run(self, plot=True, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        v_min = np.nanmin(self.data) - self.epsilon
        v_max = np.nanmax(self.data) + self.epsilon
        qspace = np.linspace(2.4, 9, self.numbins + 1) #hard coded for now.
        self.qspace = qspace
        b = np.digitize(self.data[:, :], qspace)
        b = b.reshape(1, -1)
        sumtr, trvec = count_transitions(b, self.numbins, self.lagtime)
        print("Number of transitions:", np.sum(sumtr))
        print("Transition vector:", np.sum(trvec, axis=0))
        MM = self.build_MM(sumtr, trvec, biased)
        d, v = eig(np.transpose(MM))
        mpeq = v[:, np.where(d == np.max(d))[0][0]]
        mpeq = mpeq / np.sum(mpeq)
        mpeq = mpeq.real
        rate = np.float_(- self.lagtime * conversion / np.log(d[np.argsort(d)[-2]]))
        mU2 = - self.KbT * np.log(mpeq)
        if adjust:
            mU2 -= np.min(mU2[:int(self.numbins)])
        dG = np.max(mU2[:int(self.numbins)])
        A = rate / np.exp(- dG / self.KbT) 
        x = qspace[:self.numbins]# + (qspace[1] - qspace[0])
        if plot:
            unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
            #plot the unbiased profile from 2.4 to 9 A.
            plt.plot(unb_bins, unb_profile, label="ground truth")
            plt.plot(x, mU2, label="reconstructed M by DHAMsym")
            plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.numbins))
            #plt.show()
        return x, mU2, A, MM

    def bootstrap_error(self, size, iter=100, plotall=False, save=None):
        full = self.run(plot=False)
        results = []
        data = np.copy(self.data)
        for _ in range(iter):
            idx = np.random.randint(data.shape[0], size=size)
            self.data = data[idx, :]
            try:
                results.append(self.run(plot=False, adjust=False))
            except ValueError:
                print(idx)
        r = np.array(results).astype(np.float_)
        r = r[~np.isnan(r).any(axis=(1, 2))]
        r = r[~np.isinf(r).any(axis=(1, 2))]
        if plotall:
            f, a = plt.subplots()
            for i in range(r.shape[0]):
                a.plot(r[i, 0], r[i, 1])
            plt.show()
        # interpolation
        newU = np.empty(shape=r[:, 1, :].shape, dtype=np.float_)
        for i in range(r.shape[0]):
            newU[i, :] = np.interp(full[0], r[i, 0], r[i, 1])
            # realign
            offset = align(newU[i, :], full[1])
            newU[i, :] += offset
        stderr = np.std(newU, axis=0)
        f, a = plt.subplots()
        a.plot(full[0], full[1])
        a.fill_between(full[0], full[1] - stderr, full[1] + stderr, alpha=0.2)
        plt.title("lagtime={0:d} bins={1:d}".format(self.lagtime, self.numbins))
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        self.data = data
        return
