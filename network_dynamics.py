# Project 2a - (FYP Prelim) simulated dynamics, connection extraction
# search '##' for modifiable parameters
import util
import numpy as np
from scipy.linalg import logm,inv
from scipy.stats import gaussian_kde
from time import time
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

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
        return np.sum(self.Adjacency)/(self.size*(self.size-1))

    def isBidirectional(self):
        # check if adjacency matrix is symmetric
        return np.allclose(self.Adjacency,self.Adjacency.T,0,0)

    def printAdjacency(self, file, cm=''):
        # print adjacency matrix to file
        np.savetxt(file,self.Adjacency,fmt="%d",header=cm)

    def printCoupling(self, file, cm=''):
        # print coupling matrix to file
        np.savetxt(file,self.Coupling,fmt="%.4f",header=cm)

#==============================================================================#

class network(graph):
    # CONVENTION: underscore refers to a time series
    def __init__(self):
        self.statesLog = [] # log for multiple runs

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
        beta1,beta2,y0 = 2,0.5,4 ##
        return 1/beta1*(1+np.tanh(beta2*(y-y0)))

    def couplingFuncDerivY_synaptic(self,x,y):
        # y-derivative of synaptic coupling function
        beta1,beta2,y0 = 2,0.5,4 ##
        return beta2/beta1*np.cosh(beta2*(y-y0))**-2

    def initDynamics(self, initStates, intrinsicCoef, noiseCovariance):
        # initialize node states and set noise magnitude
        self.states_ = {i:[] for i in range(self.size)}
        self.time = 0
        self.time_ = [0]
        self.iter = 1

        self.initStates = np.array(initStates)
        self.states = self.initStates
        for i in range(self.size):
            self.states_[i].append(self.states[i])

        self.intrinsicCoef = np.array(intrinsicCoef)
        self.noiseCovariance = np.array(noiseCovariance)

    def getStateChanges(self):
        # instantaneous node changes
        # changes as an np array
        WeightedCoupling = np.zeros((self.size,self.size))
        for i in range(self.size):
            WeightedCoupling[i] = self.couplingFunc_synaptic(self.states[i],self.states) ##
        WeightedCoupling *= self.Coupling

        randomVector = np.random.normal(size=self.size) ##
        # randomVector = np.random.exponential(3,size=self.size)*np.where(np.random.uniform(size=self.size)<0.5,-1,1)

        changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
            WeightedCoupling.sum(axis=1))*self.timeStep+\
            self.noiseCovariance.dot(randomVector)*np.sqrt(self.timeStep)

        return changes

    def runDynamics(self, timeStep, totIter, silent=True):
        # iterate node states according to dynamical equations
        self.timeStep = timeStep
        self.endTime = timeStep*totIter
        startTimer = time()
        while self.iter<totIter:
            self.states += self.getStateChanges()
            for i in range(self.size):
                self.states_[i].append(self.states[i])
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1

            if silent:
                print(' t = %7.2f | %c %.2f %%\r'%(self.time,util.progressBars[self.iter%4],100*self.time/self.endTime),end='')
            elif self.iter%100==0:
                print(' t = %7.2f/%7.2f | '%(self.time,self.endTime)+\
                    ' | '.join(['x%d = %7.2f'%(i,x) for i,x in enumerate(self.states)][0:min(4,self.size)]),end='')
                if self.size>4: print(' ...')
        endTimer = time()
        print('\n runDynamics() takes %d seconds'%(endTimer-startTimer))

        self.states_np = np.zeros((self.size,self.iter))
        for i in range(self.size):
            self.states_np[i] = self.states_[i]
        self.statesLog.append(self.states_)

    def removeTransient(self, transientSteps):
        # remove (initial) transient states
        # look into dynamics plot to determine what to remove
        self.iter -= transientSteps
        del self.time_[:transientSteps]
        for i in range(self.size):
            del self.states_[i][:transientSteps]

    def setSteadyStates(self):
        # set steady states with a noise-free network
        # REQUIRE: noiseCovariance = 0 (run a noise-free network separately)
        steadyStates = []
        for i in range(self.size):
            steadyStates.append(self.states_[i][-1])
        self.steadyStates = np.array(steadyStates)

    def calcTimeAvg(self):
        # compute time average of node states
        # should approx steady states
        avgStates = []
        for i in range(self.size):
            avgStates.append(np.average(self.states_[i]))
        self.avgStates = np.array(avgStates)

    def calcInfoMatrix(self):
        # compute information matrix of network (theoretical)
        # diagonal entries not usable
        print(' computing info matrix (Qij) ...')
        self.InfoMatrix = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.InfoMatrix[i] = self.couplingFuncDerivY_synaptic(self.steadyStates[i],self.steadyStates) ##
        self.InfoMatrix *= self.Coupling

    def timeCovarianceMatrix(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step
        matrixSum = 0
        for t in range(self.iter-shift):
            matrixSum += np.outer(self.states_np[:,t+shift]-self.avgStates,self.states_np[:,t]-self.avgStates)
        return matrixSum/(self.iter-shift)

    def estInfoMatrix(self):
        # estimate information matrix of network (empirical)
        # should approx InfoMatrix, compare via plotInfoMatrix
        print(' estimating info matrix (Mij) ...')
        K_0 = self.timeCovarianceMatrix(0)
        K_tau = self.timeCovarianceMatrix(1)
        self.InfoMatrix_est = logm(K_tau.dot(inv(K_0)))/self.timeStep

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

    def plotDynamics(self, file, title=None, nodes=None, color='', withSteadyStates=False):
        # plot time series data to file
        # avoid if large dataset
        print(' plotting dynamics to %s ...'%file)
        fig = plt.figure(figsize=(12,6))
        plt.xlim(np.min(self.time_),np.max(self.time_))
        if not nodes: nodes = range(self.size)
        for i in nodes:
            plt.plot(self.time_,self.states_[i],color)
            if withSteadyStates: plt.axhline(y=self.steadyStates[i],c=color,ls='--')
        if title: plt.title(title)
        plt.xlabel('time $t$')
        plt.ylabel('states $\\{x_j\\}_{1:%d}$'%self.size)
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)

    def printDynamics(self, file):
        # print time series data to file (csv format)
        # avoid if large dataset
        print(' printing dynamics to %s ...'%file)
        data = np.vstack((self.time_,self.states_np)).transpose()
        head = 't,'+','.join(map(str,range(self.size)))
        np.savetxt(file,data,delimiter=',',fmt="%.4f",header=head,comments='')
