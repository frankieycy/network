import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

class graph:
    def __init__(self, size, coupling=1, connectProb=0, adjacencyFile=None):
        # uniform bidirectional coupling
        self.size = size
        self.coupling = coupling

        if adjacencyFile: # load adjacency matrix from file
            self.loadAdjacency(adjacencyFile)
            assert(self.Adjacency.shape[0]==self.Adjacency.shape[1]==self.size)
        else: # construct random adjacency matrix
            self.Adjacency = np.zeros((self.size,self.size),dtype=int)
            self.Coupling = np.zeros((self.size,self.size))
            for i in range(self.size):
                for j in range(i+1,self.size):
                    if np.random.uniform()<connectProb:
                        self.Adjacency[i][j] = self.Adjacency[j][i] = 1

        self.Coupling = self.coupling*self.Adjacency

        # node degrees
        self.degrees = []
        for i in range(self.size):
            self.degrees.append(sum(self.Adjacency[i]))

        # Laplacian matrix
        self.laplacian = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.laplacian[i][i] = self.degrees[i]
        self.laplacian -= self.Adjacency

    def getSize(self):
        return self.size

    def getAdjacency(self):
        return self.Adjacency

    def getCoupling(self):
        return self.Coupling

    def getDegrees(self):
        return self.degrees

    def getLaplacian(self):
        return self.Laplacian

    def printAdjacency(self, file=None):
        if file:
            f = open(file, 'w')
            f.write('# adjacency matrix of size %d\n'%self.size)
            for row in self.Adjacency:
                for val in row[:-1]:
                    f.write('%d '%val)
                f.write('%d\n'%row[-1])
            f.close()
        else:
            print(self.Adjacency)

    # def printCoupling(self, file=None):
    #     if file:
    #         f = open(file, 'w')
    #         f.write('# coupling matrix of size %d\n'%self.size)
    #         for row in self.Coupling:
    #             for val in row[:-1]:
    #                 f.write('%d '%val)
    #             f.write('%d\n'%row[-1])
    #         f.close()
    #     else:
    #         print(self.Coupling)

    def loadAdjacency(self, file):
        self.Adjacency = np.loadtxt(file,dtype=int,delimiter=' ',comments='#')

    # def loadCoupling(self, file):
    #     self.Coupling = np.loadtxt(file,delimiter=' ',comments='#')

class network(graph):
    # TO DO: generalize to generic dynamics, arbitrary f & h
    def __init__(self, size, intrinsicFunc, couplingFunc, noiseSigma,\
        coupling=1, connectProb=0, adjacencyFile=None):
        super().__init__(size, coupling, connectProb, adjacencyFile)
        self.intrinsicFunc = intrinsicFunc
        self.couplingFunc = couplingFunc
        self.noiseSigma = noiseSigma

        self.initStates = None # initial states
        self.states = None # current states
        self.states_ = {i:[] for i in range(self.size)} # time series of states
        self.time = 0
        self.time_ = [0]
        self.timeStep = None
        self.iter = 1

    def getStates(self):
        return self.states

    def getDynamics(self):
        return self.states_

    def printDynamics(self, file):
        # print node states as time series to file
        f = open(file, 'w')
        f.write('time,')
        for i in range(self.size-1): f.write('%d,'%i)
        f.write('%d\n'%(self.size-1))
        for i in range(self.iter):
            f.write('%.4f,'%(i*self.timeStep))
            for j in range(self.size-1): f.write('%.4f,'%self.states_[j][i])
            f.write('%.4f\n'%self.states_[self.size-1][i])
        f.close()

    def plotDynamics(self, file):
        fig = plt.figure(figsize=(12,6))
        plt.xlim(0,self.time)
        for i in range(self.size): plt.plot(self.time_,self.states_[i])
        plt.xlabel("time $t$")
        plt.ylabel("states $\\{x_j\\}_{1:%d}$"%self.size)
        plt.grid()
        fig.tight_layout()
        fig.savefig(file)

    def initDynamics(self, initStates):
        # initialize node states
        self.initStates = np.array(initStates)
        self.states = self.initStates
        for i in range(self.size):
            self.states_[i].append(self.states[i])

    def getRates(self):
        # instantaneous node rates
        # CONSENSUS DYNAMICS, f & h implicit
        return -self.coupling*self.laplacian.dot(self.states)+\
            np.random.normal(0,self.noiseSigma,self.size)

    def runDynamics(self, timeStep, endTime):
        # evolve dynamics according to dynamical equations
        self.timeStep = timeStep
        while self.time<endTime:
            rates = self.getRates()
            self.states += self.timeStep*rates
            for i in range(self.size):
                self.states_[i].append(self.states[i])
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1
