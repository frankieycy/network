# search '##' for modifiable parameters
import util
import numpy as np
from scipy.linalg import logm,inv
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

class graph:
    def __init__(self):
        self.internalGraph = False  # graph is internally generated

    def loadGraph(self, adjacencyFile, couplingFile=None, coupling=1):
        # load adjacency matrix from file
        print(' loading adjacency file from %s ...'%adjacencyFile)
        self.Adjacency = np.loadtxt(adjacencyFile,dtype=int,delimiter=' ',comments='#')
        assert self.Adjacency.shape[0]==self.Adjacency.shape[1], 'adjacency matrix from %s not a square matrix'%file
        self.size = self.Adjacency.shape[0]

        # load coupling matrix from file
        if couplingFile:
            print(' loading coupling file from %s ...'%couplingFile)
            self.Coupling = np.loadtxt(couplingFile,delimiter=' ',comments='#')
            assert self.Coupling.shape[0]==self.Coupling.shape[1], 'coupling matrix from %s not a square matrix'%file
            assert self.Coupling.shape[0]==self.size, 'size of coupling matrix from %s does not match'%file
            self.coupling = None
        # set coupling matrix from adjacency matrix
        else:
            self.coupling = coupling
            self.Coupling = self.coupling*self.Adjacency

        self.initialize()

    def randomDirectedWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random directed graph with with Gaussian couplings
        self.internalGraph = True

        self.size = size
        self.Adjacency = np.zeros((self.size,self.size),dtype=int)
        self.Coupling = np.zeros((self.size,self.size))

        for i in range(self.size):
            for j in range(self.size):
                if np.random.uniform()<connectProb:
                    self.Adjacency[i][j] = 1
                    self.Coupling[i][j] = np.random.normal(couplingMean,couplingSpread)

        self.initialize()

    def initialize(self):
        # adjacency list
        self.AdjacencyList = {i:[] for i in range(self.size)}
        for i in range(self.size):
            for j in range(self.size):
                if self.Adjacency[i][j]:
                    self.AdjacencyList[i].append(j)

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

    def printAdjacency(self, file, cm=None):
        # print adjacency matrix to file
        f = open(file,'w')
        if cm: f.write(cm) # some notes
        f.write('# adjacency matrix of size %d\n'%self.size)
        for row in self.Adjacency:
            for val in row[:-1]:
                f.write('%d '%val)
            f.write('%d\n'%row[-1])
        f.close()

    def printCoupling(self, file, cm=None):
        # print coupling matrix to file
        f = open(file,'w')
        if cm: f.write(cm) # some notes
        f.write('# coupling matrix of size %d\n'%self.size)
        for row in self.Coupling:
            for val in row[:-1]:
                f.write('%.4f '%val)
            f.write('%.4f\n'%row[-1])
        f.close()

################################################################################

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
            WeightedCoupling[i] = self.couplingFunc_synaptic(self.states[i],self.states)
        WeightedCoupling *= self.Coupling

        randomVector = np.random.normal(size=self.size)
        # randomVector = np.random.exponential(size=self.size)*(2*(np.random.uniform(size=self.size)<0.5)-1)

        changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
            WeightedCoupling.sum(axis=1))*self.timeStep+\
            self.noiseCovariance.dot(randomVector)*np.sqrt(self.timeStep)

        return changes

    def runDynamics(self, timeStep, totIter, silent=True):
        # iterate node states according to dynamical equations
        self.timeStep = timeStep
        self.endTime = timeStep*totIter
        while self.iter<totIter:
            self.states += self.getStateChanges()
            for i in range(self.size):
                self.states_[i].append(self.states[i])
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1

            if silent:
                print(' t = %6.2f | %c %.2f %%\r'%(self.time,util.progressBars[self.iter%4],100*self.time/self.endTime),end='')
            elif self.iter%100==0:
                print(' t = %6.2f | '%self.time,end='')
                for i in range(min(self.size,4)):
                    print('x%d = %6.2f | '%(i,self.states[i]),end='')
                if self.size>4: print('...')
        self.statesLog.append(self.states_)

    def removeTransient(self, transientSteps):
        # remove (initial) transient states
        # look into dynamics plot to determine what to remove
        self.iter -= transientSteps
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
        self.InfoMatrix = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.InfoMatrix[i] = self.couplingFuncDerivY_synaptic(self.steadyStates[i],self.steadyStates)
        self.InfoMatrix *= self.Coupling

    def timeCovarianceMatrix(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step
        states_ = np.zeros((self.size,self.iter))
        for i in range(self.size):
            states_[i] = self.states_[i]

        matrixSum = 0
        for t in range(self.iter-shift):
            matrixSum += np.outer(states_[:,t]-self.avgStates,states_[:,t+shift]-self.avgStates)

        return matrixSum/(self.iter-shift)

    def estInfoMatrix(self):
        # estimate information matrix of network (empirical)
        # should approx InfoMatrix, compare via plotInfoMatrix
        return logm(self.timeCovarianceMatrix(1).dot(inv(self.timeCovarianceMatrix(0))))/self.timeStep

    def plotInfoMatrix(self, file, title=None):
        # plot information matrix: theoretical vs empirical
        Q = self.InfoMatrix
        M = self.estInfoMatrix()

        fig = plt.figure()
        for i in range(self.size):
            for j in range(self.size):
                if i!=j: plt.scatter(Q[i][j],M[i][j],s=1,c='k')
        if title: plt.title(title)
        plt.xlabel('$Q_{ij}$')
        plt.ylabel('$M_{ij}$')
        fig.tight_layout()
        fig.savefig(file)

    def plotDynamics(self, file, title=None, nodes=None):
        # plot time series data to file
        # avoid if large dataset
        print(' plotting dynamics to %s ...'%file)
        fig = plt.figure(figsize=(12,6))
        plt.xlim(0,self.time)
        if not nodes: nodes = range(self.size)
        for i in nodes: plt.plot(self.time_,self.states_[i])
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
        f = open(file, 'w')
        f.write('time,')
        for i in range(self.size-1): f.write('%d,'%i)
        f.write('%d\n'%(self.size-1))
        for t in range(self.iter):
            f.write('%.4f,'%(t*self.timeStep))
            for i in range(self.size-1): f.write('%.4f,'%self.states_[i][t])
            f.write('%.4f\n'%self.states_[self.size-1][t])
        f.close()
