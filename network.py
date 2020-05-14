import util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

class graph:
    def __init__(self):
        pass

    def load(self, adjacencyFile, couplingFile=None, coupling=1):
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

    def randomUniformGraph(self, size, connectProb, coupling=1):
        # random bidirectional graph with uniform coupling
        self.internalGraph = True # graph is internally generated

        self.size = size
        self.coupling = coupling
        self.Adjacency = np.zeros((self.size,self.size),dtype=int)

        for i in range(self.size):
            for j in range(i+1,self.size):
                if np.random.uniform()<connectProb:
                    self.Adjacency[i][j] = self.Adjacency[j][i] = 1

        self.Coupling = self.coupling*self.Adjacency

        self.initialize()

    def randomWeightedGraph(self, size, connectProb, couplingMean, couplingSpread):
        # random bidirectional graph with with Gaussian couplings
        self.internalGraph = True # graph is internally generated

        self.size = size
        self.Adjacency = np.zeros((self.size,self.size),dtype=int)

        for i in range(self.size):
            for j in range(i+1,self.size):
                if np.random.uniform()<connectProb:
                    self.Adjacency[i][j] = self.Adjacency[j][i] = 1
                    self.Coupling[i][j] = self.Coupling[j][i] = np.random.normal(couplingMean,couplingSpread)

        self.initialize()

    def initialize(self):
        # adjacency list
        self.AdjacencyList = {i:[] for i in range(self.size)}
        for i in range(self.size):
            for j in range(self.size):
                if self.Adjacency[i][j]:
                    self.AdjacencyList[i].append(j)

        # node degrees
        self.degrees = []
        for i in range(self.size):
            self.degrees.append(sum(self.Adjacency[i]))

        # Laplacian matrix
        self.laplacian = np.zeros((self.size,self.size))
        for i in range(self.size):
            self.laplacian[i][i] = self.degrees[i]
        self.laplacian -= self.Adjacency

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

    def isConnected(self):
        pass

    def shortestPaths(self, origin):
        pass

################################################################################

class network(graph):
    # CONVENTION: underscore refers to a time series
    def __init__(self):
        pass

    def loadDynamics(self, dynamicsFile):
        # load time series data from file
        print(' loading time series data from %s ...'%dynamicsFile)
        data = np.loadtxt(dynamicsFile,delimiter=',',skiprows=1) # skip header
        self.size = data.shape[1]-1
        self.time_ = data[:,0].tolist()
        self.time = self.time_[-1]
        self.timeStep = self.time_[1]-self.time_[0]
        self.iter = len(self.time_)
        self.states_ = {i:data[:,i+1].tolist() for i in range(self.size)}

    def initDynamics_Concensus(self, initStates, noiseSigma):
        # initialize node states and set noise magnitude
        self.states_ = {i:[] for i in range(self.size)}
        self.time = 0
        self.time_ = [0]
        self.iter = 1

        self.initStates = np.array(initStates)
        self.states = self.initStates
        for i in range(self.size):
            self.states_[i].append(self.states[i])

        self.noiseSigma = noiseSigma

    def getRates_Concensus(self):
        # instantaneous node rates
        # node states obey CONSENSUS DYNAMICS: dx_i/dt = -g sum_j L_ij * x_j + eta_j
        # rates as an np array
        return -self.coupling*self.laplacian.dot(self.states)+\
            np.random.normal(0,self.noiseSigma,self.size)

    def runDynamics(self, timeStep, endTime, silent=True):
        # iterate node states according to dynamical equations
        self.timeStep = timeStep
        while self.time<endTime:
            # MODIFY HERE FOR OTHER DYNAMICS
            rates = self.getRates_Concensus()
            self.states += self.timeStep*rates
            for i in range(self.size):
                self.states_[i].append(self.states[i])
            self.time += self.timeStep
            self.time_.append(self.time)
            self.iter += 1

            if silent:
                print(' t = %6.2f | %c %.2f %%\r'%(self.time,util.progressBars[self.iter%4],100*self.time/endTime),end='')
            elif self.iter%100==0:
                print(' t = %6.2f | '%self.time,end='')
                for i in range(min(self.size,4)):
                    print('x%d = %6.2f | '%(i,self.states[i]),end='')
                if self.size>4: print('...')

    def plotDynamics(self, file):
        # plot time series data to file
        # avoid if large dataset
        print(' plotting dynamics to %s ...'%file)
        fig = plt.figure(figsize=(12,6))
        plt.xlim(0,self.time)
        for i in range(self.size): plt.plot(self.time_,self.states_[i])
        plt.xlabel("time $t$")
        plt.ylabel("states $\\{x_j\\}_{1:%d}$"%self.size)
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

    def setAvgStates(self):
        # average states over all nodes at each time step
        self.avgStates_ = []
        for t in range(self.iter):
            avg = 0
            for i in range(self.size):
                avg += self.states_[i][t]
            avg /= self.size
            self.avgStates_.append(avg)

    def setCovariance(self):
        # dynamical covariance matrix from time series data
        print(' computing covariance matrix ...')
        self.setAvgStates()

        self.Covariance = np.zeros((self.size,self.size))
        states_ = {i:np.array(self.states_[i])-self.avgStates_ for i in range(self.size)} # centered about avg states
        counter = 0
        for i in range(self.size):
            for j in range(i,self.size):
                cov = states_[i].dot(states_[j])
                cov /= self.iter
                self.Covariance[i][j] = self.Covariance[j][i] = cov

                counter += 1
                print(' %c %.2f %% complete\r'%(util.progressBars[(self.size*i+j)%4],100.*counter/(self.size*(self.size+1)/2)),end='')

    def setCovarianceRatios(self):
        # covariance ratio matrix
        print(' computing covariance ratio matrix ...')
        self.CovarianceInv = np.linalg.pinv(self.Covariance) # pseudoinverse

        self.CovarianceRatios = np.zeros((self.size,self.size))
        for i in range(self.size):
            for j in range(i+1,self.size):
                # covariance ratio [i][j] = -1 / degrees [i] if i & j connected, 0 otherwise
                self.CovarianceRatios[i][j] = self.CovarianceRatios[j][i] = self.CovarianceInv[i][j]/self.CovarianceInv[i][i]

    def plotCovarianceRatios(self):
        # plot sorted covariance ratios for each node
        # may check if an apparent gap exists in each plot
        print(' plotting covariance ratios ...')
        util.setUpFolder('plt','covRatios_*.png')

        x = list(range(self.size))
        for i in range(self.size):
            y,idx = (list(t) for t in zip(*sorted(zip(self.CovarianceRatios[i], x))))
            fig = plt.figure()
            plt.xticks([])
            plt.scatter(x,y,c='k',s=10)
            for j in range(self.size):
                plt.annotate(idx[j],(x[j],y[j]))
            plt.ylabel('covariance ratio $r_{ij}$ of node $i=%d$'%i)
            fig.tight_layout()
            fig.savefig('plt/covRatios_%d.png'%i)
            plt.close()

    ############################################################################

    def estimateConnectivity(self):
        # estimate adjacency matrix from covariance ratios
        # paper: Extracting connectivity from dynamics of networks with uniform bidirectional coupling (2013)
        self.setCovariance()
        self.setCovarianceRatios()

        print(' estimating connectivity ...')

        # self.plotCovarianceRatios()

        self.AdjacencyEst = np.zeros((self.size,self.size),dtype=int)
        self.AdjacencyAcc = [0]*self.size
        for i in range(self.size):
            # sort covariance ratios and corresponding indices
            covarianceRatios = self.CovarianceRatios[i].tolist()
            idx = list(range(self.size))
            del covarianceRatios[i]
            del idx[i]
            sortedCovarianceRatios, sortedIdx = \
                (list(t) for t in zip(*sorted(zip(covarianceRatios, idx))))
            # ratio differences to search for a large ratio "gap"
            diff = np.array(sortedCovarianceRatios[1:])-np.array(sortedCovarianceRatios[:-1])

            sortedDiff = sorted(diff) # ascending
            maxDiff = sortedDiff[-1] # largest ratio difference
            secondMaxDiff = sortedDiff[-2] # second largest ratio difference
            maxDiffIdx = np.argmax(diff) # index corresponding to maxDiff

            if maxDiff>2*secondMaxDiff and self.CovarianceRatios[i][sortedIdx[maxDiffIdx]]<0:
                # when gap is sufficiently large, we are done
                connected = sortedIdx[:(maxDiffIdx+1)]
            else:
                # when gap is not sufficiently large, look for ratios that sum to -1
                cumsumCovRatios = np.cumsum(sortedCovarianceRatios)
                for j in range(len(cumsumCovRatios)):
                    if cumsumCovRatios[j]<-1:
                        maxDiffIdx = j # first index that leads to a ratio sum < -1
                        break
                # check if previous index leads to a more accurate ratio sum ~ -1
                if abs(cumsumCovRatios[maxDiffIdx-1]-(-1))<abs(cumsumCovRatios[maxDiffIdx]-(-1)):
                    maxDiffIdx -= 1
                connected = sortedIdx[:(maxDiffIdx+1)]

            self.AdjacencyAcc[i] = (diff[maxDiffIdx]/((sortedCovarianceRatios[-1]-sortedCovarianceRatios[0])/(self.size-2)))

            for j in connected: # all indices found to be connected to node i
                if self.AdjacencyAcc[i]>self.AdjacencyAcc[j]:
                    self.AdjacencyEst[i][j] = self.AdjacencyEst[j][i] = 1

    def showAnalysis(self):
        # MODIFY HERE FOR OUTPUTS
        # estimated connectivity
        print(self.AdjacencyEst)
        print(self.AdjacencyAcc)

        # performance metrics
        if self.internalGraph:
            self.trueConnected = self.Adjacency*self.AdjacencyEst
            self.trueUnconnected = (1-self.Adjacency)*(1-self.AdjacencyEst)-np.eye(self.size)
            self.sensitivity = self.trueConnected.sum(axis=1)/np.array(self.degrees)
            self.specificity = (self.trueUnconnected.sum(axis=1))/(self.size-1-np.array(self.degrees))
            self.totSensitivity = np.sum(self.trueConnected)/np.sum(self.degrees)
            self.totSpecificity = np.sum(self.trueUnconnected)/(self.size*(self.size-1)-np.sum(self.degrees))

            # print(self.sensitivity)
            # print(self.specificity)
            print(self.totSensitivity)
            print(self.totSpecificity)
