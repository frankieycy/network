import os,glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

progressBars = ['-','\\','|','/']

def setUpFolder(folder, format=None):
    # create folder if unexisting or clear folder
    if not os.path.isdir(folder):
        os.mkdir(folder)
    if format: files = glob.glob(os.path.join(folder,format))
    else: files = glob.glob(os.path.join(folder,'*'))
    for f in files:
        os.remove(f)

class graph:
    def __init__(self, adjacencyFile=None, size=0, coupling=1, connectProb=0):
        # uniform bidirectional coupling
        # (i) load adjacency matrix (requires: size, coupling, connectProb), or
        # (ii) create adjacency matrix (requires: coupling, adjacencyFile)
        self.coupling = coupling

        # adjacency matrix
        if adjacencyFile: # load adjacency matrix from file
            print(' loading adjacency file from %s ...'%adjacencyFile)
            self.loadAdjacency(adjacencyFile)
            self.size = self.Adjacency.shape[0]
        else: # create random adjacency matrix
            self.size = size
            self.Adjacency = np.zeros((self.size,self.size),dtype=int)
            for i in range(self.size):
                for j in range(i+1,self.size):
                    if np.random.uniform()<connectProb:
                        self.Adjacency[i][j] = self.Adjacency[j][i] = 1 # bidirectional

        # coupling matrix
        self.Coupling = self.coupling*self.Adjacency # uniform

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
        # print adjacency matrix to file or to screen
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
        # load adjacency matrix from file
        self.Adjacency = np.loadtxt(file,dtype=int,delimiter=' ',comments='#')
        assert self.Adjacency.shape[0]==self.Adjacency.shape[1], 'adjacency matrix from %s not a square matrix'%file

class network(graph):
    def __init__(self, dynamicsFile=None, adjacencyFile=None, size=0, coupling=1, connectProb=0, noiseSigma=0):
        # (i) load dynamics (requires: dynamicsFile, i.e., time series data)
        # (ii) generate dynamics without adjacencyFile (requires: size, coupling, connectProb, noiseSigma)
        # (iii) generate dynamics with adjacencyFile (requires: adjacencyFile, coupling, noiseSigma)
        self.avgStates_ = []
        self.Covariance = None
        self.CovarianceInv = None
        self.CovarianceRatios = None
        self.estAdjacency = None

        if dynamicsFile: # load dynamics from file
            print(' loading time series data from %s ...'%dynamicsFile)
            data = np.loadtxt(dynamicsFile,delimiter=',',skiprows=1) # skip header
            self.size = data.shape[1]-1
            self.time_ = data[:,0].tolist()
            self.time = self.time_[-1]
            self.timeStep = self.time_[1]-self.time_[0]
            self.iter = len(self.time_) # number of time steps
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
        print(' computing covariance matrix ...')
        self.setAvgStates()

        self.Covariance = np.zeros((self.size,self.size))
        avgStates_ = np.array(self.avgStates_)
        states_ = {i:np.array(self.states_[i])-avgStates_ for i in range(self.size)} # centered about avg states
        counter = 0
        for i in range(self.size):
            for j in range(i,self.size):
                cov = states_[i].dot(states_[j])
                cov /= self.iter
                self.Covariance[i][j] = self.Covariance[j][i] = cov

                counter += 1
                print(' %c %.2f %% complete\r'%(progressBars[(self.size*i+j)%4],100.*counter/(self.size*(self.size+1)/2)),end='')

    def setAvgStates(self):
        # average states over all nodes at each time step
        for t in range(self.iter):
            avg = 0
            for i in range(self.size):
                avg += self.states_[i][t]
            avg /= self.size
            self.avgStates_.append(avg)

    def estimateConnectivity(self):
        self.setCovariance()

        print(' estimating connectivity ...')
        self.CovarianceInv = np.linalg.pinv(self.Covariance)

        self.CovarianceRatios = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(i+1,self.size):
                self.CovarianceRatios[i][j] = self.CovarianceRatios[j][i] = self.CovarianceInv[i][j]/self.CovarianceInv[i][i]

        # self.plotCovarianceRatios()

        self.estAdjacency = np.zeros((self.size,self.size),dtype=int)
        for i in range(self.size):
            sortedCovarianceRatios, sortedIdx = \
                (list(t) for t in zip(*sorted(zip(self.CovarianceRatios[i], list(range(self.size))))))
            diff = np.array(sortedCovarianceRatios[1:])-np.array(sortedCovarianceRatios[:-1])

            sortedDiff = sorted(diff)
            maxDiff = sortedDiff[-1]
            secondMaxDiff = sortedDiff[-2]
            maxDiffIdx = np.argmax(diff)

            if maxDiff>2*secondMaxDiff and self.CovarianceRatios[i][sortedIdx[maxDiffIdx]]<0:
                connected = sortedIdx[:(maxDiffIdx+1)]
            else:
                cumsumCovRatios = np.cumsum(sortedCovarianceRatios)
                for j in range(len(cumsumCovRatios)):
                    if cumsumCovRatios[j]<-1:
                        maxDiffIdx = j
                        break
                if abs(cumsumCovRatios[maxDiffIdx-1]-(-1))<abs(cumsumCovRatios[maxDiffIdx]-(-1)):
                    maxDiffIdx -= 1
                connected = sortedIdx[:(maxDiffIdx+1)]

            for j in connected:
                self.estAdjacency[i][j] = 1

        print(self.estAdjacency)

    def plotCovarianceRatios(self):
        print(' plotting covariance ratios ...')
        setUpFolder('plt','covRatios_*.png')

        x = list(range(self.size))
        for i in range(self.size):
            y = sorted(self.CovarianceRatios[i])
            fig = plt.figure()
            plt.scatter(x,y,c='k')
            plt.ylabel('covariance ratio $r_{ij}$ of node $i=%d$'%i)
            fig.tight_layout()
            fig.savefig('plt/covRatios_%d.png'%i)
            plt.close()

    def printDynamics(self, file):
        # print time series data to file (csv format)
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

    def plotDynamics(self, file):
        # plot time series data to file (avoid if large dataset)
        print(' plotting dynamics to %s ...'%file)
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
        # node states obey CONSENSUS DYNAMICS: dx_i/dt = -g sum_j L_ij * x_j + eta_j
        # rates as an np array
        return -self.coupling*self.laplacian.dot(self.states)+\
            np.random.normal(0,self.noiseSigma,self.size)

    def runDynamics(self, timeStep, endTime, silent=True):
        # iterate node states according to dynamical equations
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
                print(' t = %6.2f | %c %.2f %%\r'%(self.time,progressBars[self.iter%4],100*self.time/endTime),end='')
            elif self.iter%100==0:
                print(' t = %6.2f | '%self.time,end='')
                for i in range(min(self.size,4)):
                    print('x%d = %6.2f | '%(i,self.states[i]),end='')
                if self.size>4: print('...')
