# IMPORTANT: refer to network_dynamics_cluster.py for the most updated codes
# this file is NO LONGER maintained
#==============================================================================#
# Project 2 - (FYP) neuronal network
# PyTorch handles computationally intensive calculations (for large networks ~ 10^3 nodes)
# search '##' for modifiable parameters
import util
import numpy as np
import torch
from numba import njit
from scipy.linalg import logm,inv,cholesky,eig
from scipy.stats import gaussian_kde
from scipy.sparse import csr_matrix
from time import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

@njit
def outer(x,y):
    # outer product (faster than np.outer())
    # assert x.ndim==y.ndim==1
    a = np.empty((x.size,y.size))
    for i in range(x.size):
        for j in range(y.size):
            a[i][j] = x[i]*y[j]
    return a

#==============================================================================#

class graph:
    def __init__(self):
        self.internalGraph = False  # graph is internally generated

    def loadGraph(self, couplingFile):
        # load graph (connectivity & couplings) from file
        print(' loading coupling file from %s ...'%couplingFile)
        data = np.loadtxt(couplingFile)
        indices = list(map(tuple,(data[:,0:2]-1).astype(int)))
        size = np.max(indices)+1

        self.size = size
        self.Coupling = np.zeros((size,size))
        for j in range(np.size(data,0)):
            self.Coupling[indices[j]] = data[j,2]
        self.Adjacency = (self.Coupling!=0).astype(int)

        self.initialize()

    def randomDirectedWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random directed graph with with Gaussian couplings
        self.internalGraph = True
        self.size = size
        self.Adjacency = (np.random.uniform(size=(size,size))<connectProb).astype(int)
        self.Coupling = np.random.normal(couplingMean,couplingSpread,size=(size,size))*self.Adjacency
        self.initialize()

    def randomUnidirectedWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random uni-directed graph with with Gaussian couplings
        self.internalGraph = True
        self.size = size
        self.Adjacency = np.triu((np.random.uniform(size=(size,size))<connectProb).astype(int),k=0)
        self.Coupling = np.random.normal(couplingMean,couplingSpread,size=(size,size))*self.Adjacency
        self.initialize()

    def initialize(self):
        # sparse representation
        self.sparseCoupling = csr_matrix(self.Coupling)

        # adjacency list
        self.AdjacencyList = {i:[] for i in range(self.size)}
        for i in range(self.size):
            self.AdjacencyList[i] = np.argwhere(self.Adjacency[i]==1).flatten().tolist()

        # node degrees
        self.in_degrees = []
        self.out_degrees = []
        for i in range(self.size):
            self.in_degrees.append(sum(self.Adjacency[i]))
            self.out_degrees.append(sum(self.Adjacency[:,i]))

    def calcConnectProb(self):
        # calculate empirical connection probability
        return self.Adjacency.sum()/(self.size*(self.size-1))

    def calcSparseness(self):
        # calculate sparseness of coupling matrix
        return (self.Adjacency!=0).sum()/(self.size*(self.size-1))

    def isBidirectional(self):
        # check if adjacency matrix is symmetric
        return np.allclose(self.Adjacency,self.Adjacency.T,0,0)

    def printAdjacency(self, file, cm=''):
        # print adjacency matrix to file
        np.savetxt(file,self.Adjacency,fmt='%d',header=cm)

    def printCoupling(self, file, cm=''):
        # print coupling matrix to file
        np.savetxt(file,self.Coupling,fmt='%.4f',header=cm)

#==============================================================================#

class network(graph):
    # CONVENTION: underscore refers to a time series
    def __init__(self):
        self.emailNotify = False
        self.statesLog = [] # log for multiple runs

    def setEmail(self, emailFrom, emailPw, emailTo):
        # set up email notifier
        self.emailNotify = True
        self.emailHandler = util.emailHandler(emailFrom, emailPw, emailTo)

    def intrinsicFunc(self,r,x):
        # intrinsic dynamics
        return r*x*(1-x)

    def couplingFunc_diffusive(self,x,y):
        # diffusive coupling function
        return y-x

    def couplingFuncDerivY_diffusive(self,x,y):
        # y-derivative of diffusive coupling function
        return 1

    def couplingFunc_synaptic(self,x,y):
        # synaptic coupling function
        # beta1,beta2,y0 = 2,0.5,4 ##
        beta1,beta2,y0 = 2,0.5,self.steadyStates ##
        return 1/beta1*(1+torch.tanh(beta2*(y-y0)))

    def couplingFuncDerivY_synaptic(self,x,y):
        # y-derivative of synaptic coupling function
        # beta1,beta2,y0 = 2,0.5,4 ##
        beta1,beta2,y0 = 2,0.5,self.steadyStates ##
        a = torch.cosh(beta2*(y-y0))
        return beta2/beta1/(a*a)

    #==========================================================================#

    def initDynamics(self, initStates, intrinsicCoef, noiseCovariance):
        # initialize node states and set intrinsic coef & noise cov
        print(' initializing dynamics ...')
        self.states_ = {i:[] for i in range(self.size)}
        self.time = 0
        self.time_ = [0]
        self.iter = 1

        self.initStates = np.array(initStates)
        self.states = self.initStates
        for i in range(self.size):
            self.states_[i].append(self.states[i])

        self.setIntrinsicAndNoise(intrinsicCoef,noiseCovariance)
        self.toTorch()

    def setIntrinsicAndNoise(self, intrinsicCoef, noiseCovariance):
        # set intrinsic coef & noise cov
        self.intrinsicCoef = np.array(intrinsicCoef)
        self.noiseCovariance = np.array(noiseCovariance)
        if np.allclose(self.noiseCovariance,np.zeros((self.size,self.size))):
            self.noiseChol = np.zeros((self.size,self.size))
        else:
            self.noiseChol = cholesky(self.noiseCovariance)
        if np.allclose(self.noiseChol,self.noiseChol[0,0]*np.eye(self.size)):
            self.sigma = self.noiseChol[0,0]

    def toTorch(self):
        # cast data type to torch.tensor
        self.states = torch.from_numpy(self.states)
        self.intrinsicCoef = torch.from_numpy(self.intrinsicCoef)
        self.noiseChol = torch.from_numpy(self.noiseChol)
        if isinstance(self.Coupling,np.ndarray): self.Coupling = torch.from_numpy(self.Coupling)

    #==========================================================================#

    def continueDynamics(self, file, intrinsicCoef, noiseCovariance):
        # continue dynamics from read time series data
        print(' initializing dynamics from %s ...'%file)
        self.readDynamics(file)
        self.setIntrinsicAndNoise(intrinsicCoef, noiseCovariance)
        self.toTorch()

    def readDynamics(self, file):
        # read time series data from file
        print(' reading dynamics from %s ...'%file)
        data = np.loadtxt(file,delimiter=',',skiprows=1)
        self.size = data.shape[1]-1
        self.initStates = data[0,1:]
        self.states = data[-1,1:]
        self.states_ = {i:[] for i in range(self.size)}
        for i in range(self.size):
            self.states_[i] = data[:,i+1].tolist()
        self.states_np = np.array([self.states_[i] for i in range(self.size)])
        self.time_ = data[:,0].tolist()
        self.time = self.time_[-1]
        self.timeStep = self.time_[1]-self.time_[0]
        self.sqrtTimeStep = np.sqrt(self.timeStep)
        self.iter = len(self.time_)

    #==========================================================================#

    def getStateChanges(self):
        # instantaneous node changes
        # changes as an np array
        # WeightedCoupling = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     WeightedCoupling[i] = self.couplingFunc_synaptic(self.states[i],self.states) ##
        myRow = self.couplingFunc_synaptic(None,self.states)
        WeightedCoupling = myRow.repeat(self.size,1)
        WeightedCoupling *= self.Coupling

        randomVector = torch.empty(self.size).normal_()
        # randomVector = np.random.normal(size=self.size) ##
        # randomVector = np.random.exponential(3,size=self.size)*np.where(np.random.uniform(size=self.size)<0.5,-1,1)

        # changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
        #     WeightedCoupling.sum(axis=1))*self.timeStep+\
        #     self.noiseChol.mm(randomVector)*self.sqrtTimeStep

        changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
            WeightedCoupling.sum(axis=1))*self.timeStep+\
            self.sigma*randomVector*self.sqrtTimeStep

        return changes

    def runDynamics(self, timeStep, totIter, silent=True):
        # iterate node states according to dynamical equations
        self.timeStep = timeStep
        self.sqrtTimeStep = np.sqrt(timeStep)
        self.endTime = timeStep*totIter
        startTimer = time()

        while self.iter<totIter:
            self.states += self.getStateChanges()
            for i in range(self.size):
                self.states_[i].append(self.states[i].item())
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1

            if silent:
                print(' t = %7.2f | %c %.2f %%\r'%(self.time,util.progressBars[self.iter%4],100*self.time/self.endTime),end='')
            elif self.iter%100==0:
                print(' t = %7.2f/%7.2f | '%(self.time,self.endTime)+\
                    ' | '.join(['x%d = %7.2f'%(i,x) for i,x in enumerate(self.states)][0:min(4,self.size)]),end='')
                if self.size>4: print(' ...')

            if self.emailNotify:
                if self.iter%int(totIter/5)==0:
                    self.emailHandler.sendEmail('running: t = %.2f/%.2f'%(self.time,self.endTime))

        endTimer = time()
        print('\n runDynamics() takes %.2f seconds'%(endTimer-startTimer))

        self.states_np = np.array([self.states_[i] for i in range(self.size)])
        self.statesLog.append(self.states_)

    def removeTransient(self, transientSteps):
        # remove (initial) transient states
        # look into dynamics plot to determine what to remove
        print(' removing transient states ...')
        self.iter -= transientSteps
        self.states_np = self.states_np[:,transientSteps:]
        del self.time_[:transientSteps]
        for i in range(self.size):
            del self.states_[i][:transientSteps]

    def setSteadyStates(self, file=None):
        # set steady states with a noise-free network
        # REQUIRE: noiseCovariance = 0 (run a noise-free network separately)
        print(' setting steady states ...')
        if file:
            data = np.loadtxt(file,delimiter=',',skiprows=1)
            self.steadyStates = data[1:]
        else:
            self.steadyStates = self.states_np[:,-1]

    def calcTimeAvg(self):
        # compute time average of node states
        # should approx steady states
        print(' computing time average of states ...')
        self.avgStates = self.states_np.mean(axis=1)
        if self.emailNotify: self.emailHandler.sendEmail('calcTimeAvg() completes')

    def calcInfoMatrix(self):
        # compute information matrix of network (theoretical)
        # diagonal entries not usable
        print(' computing info matrix (Qij) ...')
        # self.InfoMatrix = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     self.InfoMatrix[i] = self.couplingFuncDerivY_synaptic(self.steadyStates[i],self.steadyStates) ##
        #     util.showProgress(i+1,self.size)
        myRow = self.couplingFuncDerivY_synaptic(None,torch.from_numpy(self.steadyStates))
        self.InfoMatrix = myRow.repeat(self.size,1)
        self.InfoMatrix *= self.Coupling
        self.InfoMatrix = self.InfoMatrix.numpy()
        if self.emailNotify: self.emailHandler.sendEmail('calcInfoMatrix() completes')

    def timeCovarianceMatrix(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step
        matrixSum = 0
        for t in range(self.iter-shift):
            matrixSum += np.outer(self.states_np[:,t+shift]-self.avgStates,self.states_np[:,t]-self.avgStates)
            util.showProgress(t+1,self.iter-shift)
        return matrixSum/(self.iter-shift)

    @staticmethod
    @njit
    def timeCovarianceMatrix_fast(shift, iter, size, states_np, avgStates):
        # compute time covariance matrix (fast version)
        # shift = multiple of time step
        matrixSum = np.zeros((size,size))
        for t in range(iter-shift):
            matrixSum += outer(states_np[:,t+shift]-avgStates,states_np[:,t]-avgStates)
        return matrixSum/(iter-shift)

    def estInfoMatrix(self):
        # estimate information matrix of network (empirical)
        # should approx InfoMatrix, compare via plotInfoMatrix
        print(' estimating info matrix (Mij) ...')
        K_0 = self.timeCovarianceMatrix_fast(0,self.iter,self.size,self.states_np,self.avgStates)
        K_tau = self.timeCovarianceMatrix_fast(1,self.iter,self.size,self.states_np,self.avgStates)
        print(' estimating info matrix (Mij) ... taking logm')
        self.InfoMatrix_est = logm(K_tau.dot(inv(K_0)))/self.timeStep
        if self.emailNotify: self.emailHandler.sendEmail('estInfoMatrix() completes')

    def checkLogmCondition(self):
        # check if condition that log(exp(Q))=Q is satisfied
        print(' checking if condition that log(exp(Q))=Q is satisfied ...')
        eigVal,_ = eig(self.InfoMatrix)
        print(' - condition (tau max_i |Im λ_i| < π) is',
            self.timeStep*np.max(np.imag(eigVal))<np.pi)

    def plotInfoMatrix(self, file, title=None):
        # plot information matrix: theoretical vs empirical
        # REQUIRE: calcInfoMatrix() and estInfoMatrix() beforehand
        print(' plotting info matrix (Qij vs Mij) to %s ...'%file)
        Q = self.InfoMatrix
        M = self.InfoMatrix_est
        Q = Q[~np.eye(Q.shape[0],dtype=bool)].reshape(Q.shape[0],-1).flatten()
        M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1).flatten()

        fig = plt.figure()
        plt.scatter(Q,M,s=1,c='k')
        if title: plt.title(title)
        plt.xlabel('$Q_{ij}$')
        plt.ylabel('$M_{ij}$')
        fig.tight_layout()
        fig.savefig(file)

    def plotEstInfoMatrix(self, file, title=None):
        # plot estimated information matrix: distribution
        print(' plotting estimated info matrix (Mij distribution) to %s ...'%file)
        M = self.InfoMatrix_est
        M = M[~np.eye(M.shape[0],dtype=bool)].reshape(M.shape[0],-1).flatten()
        mean = np.mean(M)
        sd = np.std(M)

        fig = plt.figure()
        x = np.linspace(np.min(M),np.max(M),100)
        density = gaussian_kde(M)
        plt.plot(x,density(x),'k',label='$\sigma=%.1f$'%sd)
        plt.scatter(M,[0]*len(M),s=5,c='k')
        plt.axvline(x=-sd,c='k',ls='--')
        plt.axvline(x=+sd,c='k',ls='--')
        plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('$M_{ij}$')
        plt.legend()
        fig.tight_layout()
        fig.savefig(file)

    def plotDynamics(self, file, title=None, nodes=None, iterSlice=None, color='', withSteadyStates=False):
        # plot time series data to file
        print(' plotting dynamics to %s ...'%file)
        fig = plt.figure(figsize=(12,6))
        if not nodes: nodes = range(self.size)
        if not iterSlice: iterSlice = slice(self.iter)
        plt.xlim(self.time_[iterSlice][0],self.time_[iterSlice][-1])
        for i in nodes:
            plt.plot(self.time_[iterSlice],self.states_np[i,iterSlice],color)
            if withSteadyStates: plt.axhline(y=self.steadyStates[i],c=color,ls='--')
        if title: plt.title(title)
        plt.xlabel('time $t$')
        plt.ylabel('states $\\{x_j\\}_{1:%d}$'%self.size)
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)

    def printDynamics(self, file, iterSlice=None, lastSlice=False):
        # print time series data to file (csv format)
        print(' printing dynamics to %s ...'%file)
        if not iterSlice: iterSlice = slice(self.iter)
        if lastSlice:
            data = np.concatenate(([self.time_[-1]],self.states_np[:,-1])).reshape((1,self.size+1))
        else:
            data = np.vstack((self.time_[iterSlice],self.states_np[iterSlice])).transpose()
        head = 't,'+','.join(map(str,range(self.size)))
        np.savetxt(file,data,delimiter=',',fmt='%.4f',header=head,comments='')
