import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

progressBars = ['-','\\','|','/']

class graph:
    def __init__(self, adjacencyFile=None, size=0, coupling=1, connectProb=0):
        # uniform bidirectional coupling
        # (i) load adjacency matrix (require size, coupling, connectProb), or
        # (ii) create adjacency matrix (require coupling, adjacencyFile)
        self.coupling = coupling

        # adjacency matrix
        if adjacencyFile: # load adjacency matrix from file
            print('loading adjacency file from %s...'%adjacencyFile)
            self.loadAdjacency(adjacencyFile)
            self.size = self.Adjacency.shape[0]
        else: # create random adjacency matrix
            self.size = size
            self.Adjacency = np.zeros((self.size,self.size),dtype=int)
            for i in range(self.size):
                for j in range(i+1,self.size):
                    if np.random.uniform()<connectProb:
                        self.Adjacency[i][j] = self.Adjacency[j][i] = 1

        # coupling matrix
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
        return self.laplacian

    def printAdjacency(self, file=None):
        if file:
            f = open(file,'w')
            f.write('# adjacency matrix of size %d\n'%self.size)
            for row in self.Adjacency:
                for val in row[:-1]:
                    f.write('%d '%val)
                f.write('%d\n'%row[-1])
            f.close()
        else:
            print(m)

    def loadAdjacency(self, file):
        self.Adjacency = np.loadtxt(file,dtype=int,delimiter=' ',comments='#')
        assert self.Adjacency.shape[0]==self.Adjacency.shape[1], 'adjacency matrix from %s not a square matrix'%file

class network(graph):
    def __init__(self, dynamicsFile=None, adjacencyFile=None, size=0, coupling=1, connectProb=0, noiseSigma=0):
        # (i) load dynamics (require dynamicsFile)
        # (ii) generate dynamics without adjacencyFile (require size, coupling, connectProb, noiseSigma)
        # (iii) generate dynamics with adjacencyFile (require adjacencyFile, coupling, noiseSigma)
        self.avgStates_ = []
        self.Covariance = None
        self.CovarianceInv = None
        self.CovarianceRatios = None

        if dynamicsFile: # load dynamics from file
            print('loading dynamics file from %s...'%dynamicsFile)
            data = np.loadtxt(dynamicsFile,delimiter=',',skiprows=1)
            self.size = data.shape[1]-1
            self.time_ = data[:,0].tolist()
            self.time = self.time_[-1]
            self.timeStep = self.time_[1]-self.time_[0]
            self.iter = len(self.time_)
            self.states_ = {i:data[:,i+1].tolist() for i in range(self.size)}

        else: # generate dynamics
            super().__init__(adjacencyFile, size, coupling, connectProb)
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

    def setCovariance(self):
        print('computing covariance matrix...')
        self.setAvgStates()

        self.Covariance = np.zeros((self.size,self.size))
        counter = 0
        # IMPROVE: speed up the computation!
        for i in range(self.size):
            for j in range(i,self.size):
                cov = 0
                for t in range(self.iter):
                    cov += (self.states_[i][t]-self.avgStates_[t])*(self.states_[j][t]-self.avgStates_[t])
                cov /= self.iter
                self.Covariance[i][j] = self.Covariance[j][i] = cov

                counter += 1
                print('%c %.2f %% complete\r'%(progressBars[(self.size*i+j)%4],100.*counter/(self.size*(self.size+1)/2)),end='')

    def setAvgStates(self):
        for t in range(self.iter):
            avg = 0
            for i in range(self.size):
                avg += self.states_[i][t]
            avg /= self.size
            self.avgStates_.append(avg)

    def estimateConnectivity(self):
        self.setCovariance()

        print('estimating connectivity...')
        self.CovarianceInv = np.linalg.pinv(self.Covariance)

        self.CovarianceRatios = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(i+1,self.size):
                self.CovarianceRatios[i][j] = self.CovarianceRatios[j][i] = self.CovarianceInv[i][j]/self.CovarianceInv[i][i]

        # print(self.Covariance[0])
        # print(self.CovarianceInv[0])
        print(self.CovarianceRatios[0])

        plt.scatter(list(range(self.size)),sorted(self.CovarianceRatios[0]))
        plt.show()

    def printDynamics(self, file):
        # print node states as time series to file
        print('printing dynamics to %s...'%file)
        f = open(file, 'w')
        f.write('time,')
        for i in range(self.size-1): f.write('%d,'%i)
        f.write('%d\n'%(self.size-1))
        for t in range(self.iter):
            f.write('%.4f,'%(t*self.timeStep))
            for i in range(self.size-1): f.write('%.4f,'%self.states_[i][t])
            f.write('%.4f\n'%self.states_[self.size-1][t])
        f.close()

    def plotDynamics(self, file):
        print('plotting dynamics to %s...'%file)
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
        # CONSENSUS DYNAMICS: dx_i/dt = -g sum_j L_ij * x_j + eta_j
        return -self.coupling*self.laplacian.dot(self.states)+\
            np.random.normal(0,self.noiseSigma,self.size)

    def runDynamics(self, timeStep, endTime, silent=True):
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

            if silent:
                print('t = %6.2f | %c %.2f %%\r'%(self.time,progressBars[self.iter%4],100*self.time/endTime),end='')
            elif self.iter%100==0:
                print('t = %6.2f | '%self.time,end='')
                for i in range(min(self.size,4)):
                    print('x%d = %6.2f | '%(i,self.states[i]),end='')
                if self.size>4: print('...')
