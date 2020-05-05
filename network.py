import numpy as np
import matplotlib.pyplot as plt

class graph:
    def __init__(self, size, connectProb=0, coupling=0, adjacencyFile=None, couplingFile=None):
        self.size = size
        self.degrees = None
        self.Laplacian = None
        if adjacencyFile and couplingFile:
            # load matrices from files
            self.loadAdjacency(adjacencyFile)
            self.loadCoupling(couplingFile)
            assert(self.Adjacency.shape[0]==self.Adjacency.shape[1]==\
                  self.Coupling.shape[0]==self.Coupling.shape[1]==self.size)
        else:
            self.Adjacency = np.zeros((self.size,self.size))
            self.Coupling = np.zeros((self.size,self.size))
            for i in range(self.size):
                for j in range(i+1,self.size):
                    if np.random.uniform()<connectProb:
                        # symmetric matrices for uniform bidirectional coupling
                        self.Adjacency[i][j] = self.Adjacency[j][i] = 1
                        self.Coupling[i][j] = self.Coupling[j][i] = coupling

    def getSize(self):
        return self.size
    def getAdjacency(self):
        return self.Adjacency
    def getCoupling(self):
        return self.Coupling
    def getDegrees(self):
        # node degrees
        if self.degrees: return self.degrees
        degrees = []
        for i in range(self.size):
            degrees.append(sum(self.Adjacency[i]))
        self.degrees = degrees
        return self.degrees
    def getLaplacian(self):
        # Laplacian matrix
        # TO DO: incorporate couplings
        if self.Laplacian: return self.Laplacian
        laplacian = np.zeros((self.size,self.size))
        degrees = self.getDegrees()
        for i in range(self.size):
            laplacian[i][i] = degrees[i]
        self.Laplacian = laplacian-self.Adjacency
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
    def printCoupling(self, file=None):
        if file:
            f = open(file, 'w')
            f.write('# coupling matrix of size %d\n'%self.size)
            for row in self.Coupling:
                for val in row[:-1]:
                    f.write('%d '%val)
                f.write('%d\n'%row[-1])
            f.close()
        else:
            print(self.Coupling)
    def printGraph(self):
        pass

    def loadAdjacency(self, file):
        self.Adjacency = np.loadtxt(file,delimiter=' ',comments='#')
    def loadCoupling(self, file):
        self.Coupling = np.loadtxt(file,delimiter=' ',comments='#')

class network(graph):
    def __init__(self, size, intrinsicFunc, couplingFunc, noiseSigma,\
        connectProb=0, coupling=0, adjacencyFile=None, couplingFile=None):
        super().__init__(size, connectProb, coupling, adjacencyFile, couplingFile)
        self.intrinsicFunc = intrinsicFunc
        self.couplingFunc = couplingFunc
        self.noiseSigma = noiseSigma

        self.initStates = None # initial states
        self.states = None # current states
        self.dynamics = {i:[] for i in range(self.size)} # time series of states
        self.times = [0]
        self.time = 0
        self.timeStep = None
        self.iter = 1

    def initDynamics(self, initStates):
        # initialize node values
        self.initStates = initStates
        self.states = np.array(self.initStates)
        for i in range(self.size):
            self.dynamics[i].append(self.initStates[i])

    def nodeRate(self):
        # TO DO: an np array of rates for all nodes
        pass

    def runDynamics(self, timeStep, endTime):
        # evolve dynamics according to dynamical equations
        # TO DO: update states through np matrix multiplication
        # log states to screen when running
        self.timeStep = timeStep
        while self.time<endTime:
            # 
            self.time += self.timeStep
            self.times.append(self.time)
            self.iter += 1
            pass

    def getDynamics(self):
        return self.dynamics

    def printDynamics(self, file):
        # print node values as time series in csv format
        f = open(file, 'w')
        f.write('time,')
        for i in range(self.size-1): f.write('%d,'%i)
        f.write('%d\n'%(self.size-1))
        #
        f.close()
        pass

def f(x):
    return 0

def h(xi,xj):
    return xi-xj
