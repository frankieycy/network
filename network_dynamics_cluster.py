# Project 2a - (FYP Prelim) simulated dynamics, connection extraction
# numba handles computationally intensive calculations
# for use in phy dept clusters (this is the main file to edit)
# search '##' for modifiable model parameters
import util
import istarmap
import numpy as np
from numba import njit,prange
from scipy.linalg import logm,inv,cholesky,eig
from scipy.stats import gaussian_kde,norm
from scipy.sparse import csr_matrix
from time import time
from tqdm import tqdm
from multiprocessing import Pool,cpu_count
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
plt.switch_backend('agg')

@njit
def normalVector(mean=0, spread=1, size=1):
    # vector of normal random numbers (faster than np.random.normal())
    a = np.empty(size)
    for i in range(size):
        a[i] = np.random.normal(mean,spread)
    return a

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

        # node degrees
        self.degrees_in = self.Adjacency.sum(axis=1)
        self.degrees_out = self.Adjacency.sum(axis=0)
        with np.errstate(invalid='ignore'):
            self.strengths_in = np.nan_to_num(self.Coupling.sum(axis=1)/self.degrees_in)
            self.strengths_out = np.nan_to_num(self.Coupling.sum(axis=0)/self.degrees_out)

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

    @staticmethod
    @njit
    def intrinsicFunc(r,x):
        # intrinsic dynamics
        return r*x*(1-x)

    @staticmethod
    @njit
    def intrinsicFunc_fast(size,r,x):
        # intrinsic dynamics (fast version)
        a = np.empty(size)
        for i in range(size):
            a[i] = r[i]*x[i]*(1-x[i])
        return a

    @staticmethod
    @njit
    def couplingFunc_diffusive(x,y):
        # diffusive coupling function
        return y-x

    @staticmethod
    @njit
    def couplingFuncDerivY_diffusive(x,y):
        # y-derivative of diffusive coupling function
        return 1

    @staticmethod
    @njit
    def couplingFunc_synaptic(x,y):
        # synaptic coupling function
        beta1,beta2,y0 = 20,1,1 ##
        return 1/beta1*(1+np.tanh(beta2*(y-y0)))

    @staticmethod
    @njit
    def couplingFuncDerivY_synaptic(x,y):
        # y-derivative of synaptic coupling function
        beta1,beta2,y0 = 20,1,1 ##
        a = np.cosh(beta2*(y-y0))
        return beta2/beta1/(a*a)

    # initialization ==========================================================#

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

    # initialization from file ================================================#

    def continueDynamics(self, file, intrinsicCoef, noiseCovariance):
        # continue dynamics from read time series data
        print(' initializing dynamics from %s ...'%file)
        self.readDynamics(file)
        self.setIntrinsicAndNoise(intrinsicCoef, noiseCovariance)

    def readDynamics(self, file):
        # read time series data from file
        print(' reading dynamics from %s ...'%file)
        if isinstance(file,list):
            # p = Pool(cpu_count()-1)
            # if file[0][-4:]=='.npy': data = np.vstack(p.map(np.load,file)) # [npy files]
            # else: data = np.vstack(p.map(util.loadcsv,file)) # [csv files]
            if file[0][-4:]=='.npy': data = np.vstack([np.load(f) for f in file]) # [npy files]
            else: data = np.vstack([util.loadcsv(f) for f in file]) # [csv files]
        else:
            if file[-4:]=='.npy': data = np.load(file) # npy file
            else: data = util.loadcsv(file) # csv file
        print(' finished reading dynamics ...')

        self.size = data.shape[1]-1
        self.initStates = data[0,1:]
        self.states = data[-1,1:]

        if file[:6]=='(cont)': # cont file for continuation (contains only two time steps)
            self.states_ = {i:[] for i in range(self.size)}
            self.time_ = []
        else: # not cont file
            # self.states_ = {i:data[:,i+1].tolist() for i in range(self.size)} # in most cases, this line is optional
            self.time_ = data[:,0].tolist()

        self.states_np = data[:,1:].T
        self.time = data[-1,0]
        self.timeStep = data[1,0]-data[0,0]
        self.sqrtTimeStep = np.sqrt(self.timeStep)
        self.iter = int(self.time/self.timeStep)

    def saveNpDynamics(self, file):
        # print time series data to file (npy format)
        # npy file for fast loading
        print(' printing dynamics to %s ...'%file)
        np.save(file,np.vstack((self.time_,self.states_np)).T)

    # generate dynamics =======================================================#

    def getStateChanges(self):
        # instantaneous node changes
        # changes as an np array

        # WeightedCoupling = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     WeightedCoupling[i] = self.couplingFunc_synaptic(self.states[i],self.states) ##
        # WeightedCoupling *= self.Coupling

        # using feature of synaptic coupling function
        myRow = self.couplingFunc_synaptic(None,self.states)
        # WeightedCoupling = np.multiply(self.Coupling,myRow)
        WeightedCoupling = self.sparseCoupling.multiply(myRow)

        randomVector = np.random.normal(size=self.size) ##
        # randomVector = np.random.exponential(3,size=self.size)*np.where(np.random.uniform(size=self.size)<0.5,-1,1)

        # changes = (self.intrinsicFunc(self.intrinsicCoef,self.states)+\
        #     WeightedCoupling.sum(axis=1))*self.timeStep+\
        #     self.noiseChol.dot(randomVector)*self.sqrtTimeStep

        # changes = (self.intrinsicFunc_fast(self.size,self.intrinsicCoef,self.states)+\
        #     WeightedCoupling.sum(axis=1))*self.timeStep+\
        #     self.sigma*randomVector*self.sqrtTimeStep
        #     # self.sigma*normalVector(size=self.size)*self.sqrtTimeStep

        changes = (self.intrinsicFunc_fast(self.size,self.intrinsicCoef,self.states)+\
            WeightedCoupling.sum(axis=1).A1)*self.timeStep+\
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

            if self.emailNotify:
                if self.iter%(totIter//5)==0:
                    self.emailHandler.sendEmail('running: t = %.2f/%.2f'%(self.time,self.endTime))

        endTimer = time()
        print('\n runDynamics() takes %.2f seconds'%(endTimer-startTimer))

        self.states_np = np.array([self.states_[i] for i in range(self.size)])
        self.statesLog.append(self.states_)

    # post-dynamics processing ================================================#

    def removeTransient(self, transientSteps):
        # remove (initial) transient states
        # look into dynamics plot to determine what to remove
        print(' removing transient states ...')
        self.iter -= transientSteps
        self.states_np = self.states_np[:,transientSteps:]
        del self.time_[:transientSteps]
        # for i in range(self.size):
        #     del self.states_[i][:transientSteps]

    def setSteadyStates(self, file=None):
        # set steady states with a noise-free network
        # or load steady states from file
        # REQUIRE: noiseCovariance = 0 (run a noise-free network separately)
        print(' setting steady states ...')
        if file:
            data = np.loadtxt(file,delimiter=',',skiprows=1)
            self.steadyStates = data[1:]
        else:
            self.steadyStates = self.states_np[:,-1] # last time step

    def calcTimeAvg(self):
        # compute time average of node states
        # should approx steady states
        print(' computing time average of states ...')
        self.avgStates = self.states_np.mean(axis=1)
        if self.emailNotify: self.emailHandler.sendEmail('calcTimeAvg() completes')

    def setTimeAvg(self, file):
        # load time average of node states from file
        print(' setting time average of states from %s ...'%file)
        self.avgStates = np.load(file)

    def printTimeAvg(self, file):
        # print time average of node states to file
        print(' printing time average of states to %s ...'%file)
        np.save(file,self.avgStates)

    def calcStatesFluc(self):
        # compute mean & s.d. of fluctuations around steady states
        print(' computing mean & s.d. of fluctuations around steady states ...')
        self.statesFluc = self.states_np-self.steadyStates.reshape(-1,1) # fluc ard steady states
        self.flucMean = self.statesFluc.mean(axis=1)
        self.flucSd = self.statesFluc.std(axis=1)
        if self.emailNotify: self.emailHandler.sendEmail('calcStatesFluc() completes')

    # analysis ================================================================#

    def calcInfoMatrix(self):
        # compute information matrix of network (theoretical)
        # diagonal entries not usable
        print(' computing info matrix (Qij) ...')

        # self.InfoMatrix = np.empty((self.size,self.size))
        # for i in range(self.size):
        #     self.InfoMatrix[i] = self.couplingFuncDerivY_synaptic(self.steadyStates[i],self.steadyStates) ##
        #     util.showProgress(i+1,self.size)

        # using feature of synaptic coupling function
        myRow = self.couplingFuncDerivY_synaptic(None,self.steadyStates)
        # self.InfoMatrix = np.multiply(self.Coupling,myRow)
        self.InfoMatrix = self.sparseCoupling.multiply(myRow).A

        if self.emailNotify: self.emailHandler.sendEmail('calcInfoMatrix() completes')

    def timeCovarianceMatrix(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step

        # matrixSum = 0
        # for t in range(self.iter-shift):
        #     matrixSum += outer(self.states_np[:,t+shift]-self.avgStates,self.states_np[:,t]-self.avgStates)
        #     util.showProgress(t+1,self.iter-shift)

        matrixSum = 0
        statesFluc = self.states_np.T-self.avgStates # fluc ard avg states
        _ = np.empty((self.size,self.size)) # for faster np.outer()

        for t in tqdm(range(self.iter-shift)):
            matrixSum += np.outer(statesFluc[t+shift],statesFluc[t],_)

        return matrixSum/(self.iter-shift)

    def timeCovarianceMatrix_parallel(self, shift=0):
        # compute time covariance matrix
        # shift = multiple of time step

        p = Pool(cpu_count()-1)
        matrixSum = 0
        statesFluc = self.states_np.T-self.avgStates
        _ = np.empty((self.size,self.size))

        if shift==0: for myOuter in tqdm(p.istarmap(np.outer,zip(statesFluc,statesFluc,[_]*self.iter)),total=self.iter): matrixSum += myOuter
        else: for myOuter in tqdm(p.starmap(np.outer,zip(statesFluc[:-shift],statesFluc[shift:],[_]*self.iter)),total=self.iter-shift): matrixSum += myOuter

        return matrixSum/(self.iter-shift)

    @staticmethod
    @njit(parallel=True,nogil=True)
    def timeCovarianceMatrix_fast(shift, iter, size, states_np, avgStates):
        # compute time covariance matrix (fast version)
        # shift = multiple of time step

        statesFluc = states_np.T-avgStates
        matrixSum = np.zeros((size,size))
        _ = np.empty((size,size))

        for t in prange(iter-shift):
            matrixSum += np.outer(statesFluc[t+shift],statesFluc[t],_)

        return matrixSum/(iter-shift)

    def estInfoMatrix(self):
        # estimate information matrix of network (empirical)
        # should approx InfoMatrix, compare via plotInfoMatrix
        print(' estimating info matrix (Mij) ...')

        K_0 = self.timeCovarianceMatrix(0)
        K_tau = self.timeCovarianceMatrix(1)

        # K_0 = self.timeCovarianceMatrix_parallel(0)
        # K_tau = self.timeCovarianceMatrix_parallel(1)

        # K_0 = self.timeCovarianceMatrix_fast(0,self.iter,self.size,self.states_np,self.avgStates)
        # K_tau = self.timeCovarianceMatrix_fast(1,self.iter,self.size,self.states_np,self.avgStates)

        print(' estimating info matrix (Mij) ... taking logm')
        self.InfoMatrix_est = logm(K_tau.dot(inv(K_0)))/self.timeStep

        if self.emailNotify: self.emailHandler.sendEmail('estInfoMatrix() completes')

    def checkLogmCondition(self):
        # check if condition that log(exp(Q))=Q is satisfied
        print(' checking if condition that log(exp(Q))=Q is satisfied ...')
        eigVal,_ = eig(self.InfoMatrix)
        print(' - condition (tau max_i |Im lambda_i| < pi) is',
            self.timeStep*np.max(np.imag(eigVal))<np.pi)

    # plots ===================================================================#

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
        x = np.linspace(np.min(M),np.max(M),200)
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

    def plotDegreeStengthDist(self, degreeFile, strengthFile, degreeTitle=None, strengthTitle=None):
        # plotting degree and strength distribution
        print(' plotting degree & strength distribution to %s & %s ...'%(degreeFile,strengthFile))

        fig = plt.figure()
        minDeg = np.percentile((self.degrees_in,self.degrees_out),0)
        maxDeg = np.percentile((self.degrees_in,self.degrees_out),98)
        x = np.linspace(minDeg,maxDeg,200)
        density_in = gaussian_kde(self.degrees_in)
        density_out = gaussian_kde(self.degrees_out)
        plt.plot(x,density_in(x),'k',label='in-degrees')
        plt.plot(x,density_out(x),'k--',label='out-degrees')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        if degreeTitle: plt.title(degreeTitle)
        plt.xlabel('degrees $k_i$')
        plt.legend()
        # fig.tight_layout()
        fig.savefig(degreeFile)

        fig = plt.figure()
        minStren = np.percentile((self.strengths_in,self.strengths_out),2)
        maxStren = np.percentile((self.strengths_in,self.strengths_out),98)
        x = np.linspace(minStren,maxStren,200)
        density_in = gaussian_kde(self.strengths_in)
        density_out = gaussian_kde(self.strengths_out)
        plt.plot(x,density_in(x),'k',label='in-strengths')
        plt.plot(x,density_out(x),'k--',label='out-strengths')
        plt.ylim(bottom=0)
        if strengthTitle: plt.title(strengthTitle)
        plt.xlabel('strengths $s_i$')
        plt.legend()
        # fig.tight_layout()
        fig.savefig(strengthFile)

    def plotFlucDist(self, file, nodes, title=None):
        # plot distribution of fluctuations around steady states
        print(' plotting fluctuation distribution (xi-Xi distribution) to %s ...'%file)

        fig = plt.figure()
        minFluc = np.percentile(self.statesFluc[nodes],2)
        maxFluc = np.percentile(self.statesFluc[nodes],98)
        x = np.linspace(minFluc,maxFluc,200)
        colors = list(map(tuple,np.random.rand(self.size,3)))
        for i in nodes:
            density = gaussian_kde(self.statesFluc[i])
            mean = self.statesFluc[i].mean()
            sd = self.statesFluc[i].std()
            plt.plot(x,density(x),label='node %d: $\mu=%.2f,\sigma=%.2f$'%(i,mean,sd),c=colors[i])
            plt.plot(x,norm(loc=mean,scale=sd).pdf(x),c=colors[i],ls='--')
        plt.ylim(bottom=0)
        if title: plt.title(title)
        plt.xlabel('fluctuation $x_i-X_i$')
        plt.legend()
        # fig.tight_layout()
        fig.savefig(file)

    def plotFlucSdAgainstDegStren(self, degreeFile, strengthFile, degreeTitle=None, strengthTitle=None):
        # plot s.d. of fluctuations against degrees & strengths
        print(' plotting s.d. of fluctuations against degrees & strengths to %s & %s ...'%(degreeFile,strengthFile))

        fig = plt.figure()
        plt.scatter(self.degrees_in,self.flucSd,label='in-degrees',s=1,c='b')
        plt.scatter(self.degrees_out,self.flucSd,label='out-degrees',s=1,c='r')
        if degreeTitle: plt.title(degreeTitle)
        plt.xlabel('degrees')
        plt.ylabel('s.d. of fluctuation')
        plt.legend()
        fig.tight_layout()
        fig.savefig(degreeFile)

        fig = plt.figure()
        plt.scatter(self.strengths_in,self.flucSd,label='in-strengths',s=1,c='b')
        plt.scatter(self.strengths_out,self.flucSd,label='out-strengths',s=1,c='r')
        if degreeTitle: plt.title(degreeTitle)
        plt.xlabel('strengths')
        plt.ylabel('s.d. of fluctuation')
        plt.legend()
        fig.tight_layout()
        fig.savefig(strengthFile)

    def plotDynamics(self, file, title=None, nodes=None, iterSlice=None, color=None, ylimRange=None, withSteadyStates=False):
        # plot time series data to file
        print(' plotting dynamics to %s ...'%file)

        fig = plt.figure(figsize=(12,6))
        if not nodes: nodes = range(self.size)
        if not iterSlice: iterSlice = slice(self.iter)

        plt.xlim(self.time_[iterSlice][0],self.time_[iterSlice][-1])
        if ylimRange: plt.ylim(ylimRange)

        if color: colors = [color]*self.size
        else: colors = list(map(tuple,np.random.rand(self.size,3)))

        for i in nodes:
            plt.plot(self.time_[iterSlice],self.states_np[i,iterSlice],c=colors[i])
            if withSteadyStates: plt.axhline(y=self.steadyStates[i],c=colors[i],ls='--')

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
        if lastSlice: data = np.concatenate(([self.time_[-1]],self.states_np[:,-1])).reshape((1,self.size+1))
        else: data = np.vstack((self.time_[iterSlice],self.states_np[:,iterSlice])).T

        head = 't,'+','.join(map(str,range(self.size)))
        np.savetxt(file,data,delimiter=',',fmt='%.4f',header=head,comments='')

    def printContFile(self, file):
        # print time series data to cont file (csv format)
        self.printDynamics('(cont)'+file,iterSlice=slice(-2,None))
