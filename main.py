# this file does the actual work
# useful code sections are cached below
import numpy as np
import matplotlib.pyplot as plt
import network_dynamics_cluster as n
# np.random.seed(0)

myNet = n.network()
# myNet.loadGraph('DIV25_PREmethod')
# myNet.loadNpGraph('')

# myNet.readPeakTime('(neuron_FHN_synaptic_withNoise_sigma=2_e=0.1_a=0.95_0.01,1,0)PeakTime_t=0to1000_h=0.00_d=2000.txt',withIdx=True)
myNet.readPeakTime('updatedDIV25_spks',withIdx=False)
myNet.makeCoupling_FNCCH(10000) # use W=10000
# np.save('DIV25_FNCCH.npy',myNet.Coupling)

# f1 = open('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=0.95)PeakTime_t=0to500_h=0.00_d=2000.txt')
# f2 = open('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=0.95)PeakTime_t=500to1000_h=0.00_d=2000.txt')
# l1 = f1.readlines()
# l2 = f2.readlines()
#
# d1 = {i:l1[i].rstrip('\n').split()[2:] for i in range(4095)}
# d2 = {i:l2[i].rstrip('\n').split()[2:] for i in range(4095)}
# for i in range(4095):
#     for j,x in enumerate(d2[i]): d2[i][j] = str(10**6+int(x))
#     d1[i].extend(d2[i])
#
# myNet.size = 4095
# myNet.peakDict = d1
# myNet.printPeakTime('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=0.95)PeakTime_t=0to1000_h=0.00_d=2000.txt')

#==============================================================================#
#### read in peak count files (different dynamics) and compare distribution plots #### AD HOC

# a = np.load('')
# b = np.load('')
# c = np.load('')
# a = (a-a.mean())/a.std()
# b = (b-b.mean())/b.std()
# c = (c-c.mean())/c.std()
#
# from scipy.stats import norm,gaussian_kde
# fig = plt.figure()
# x = np.linspace(-4,4,200)
# plt.plot(x,norm(loc=0,scale=1).pdf(x),'r--')
# density_a = gaussian_kde(a)
# density_b = gaussian_kde(b)
# density_c = gaussian_kde(c)
# plt.plot(x,density_a(x),label='')
# plt.plot(x,density_b(x),label='')
# plt.plot(x,density_c(x),label='')
# plt.xlabel('peak count')
# plt.xlim(left=-4,right=4)
# plt.legend()
# fig.tight_layout()
# fig.savefig('')
# plt.close()

#==============================================================================#
#### read in peak count files (different time steps) and compare distribution plots #### AD HOC

# from scipy.stats import norm,gaussian_kde
# a = np.load('')
# b = np.load('')
# a = (a-a.mean())/a.std()
# b = (b-b.mean())/b.std()
#
# fig = plt.figure()
# x = np.linspace(-4,4,200)
# plt.plot(x,norm(loc=0,scale=1).pdf(x),'r--')
# density_a = gaussian_kde(a)
# density_b = gaussian_kde(b)
# plt.plot(x,density_a(x),label='')
# plt.plot(x,density_b(x),label='')
# plt.xlabel('peak count')
# plt.legend()
# fig.tight_layout()
# fig.savefig('')
# plt.close()

#==============================================================================#
#### read in peak count file and print to csv #### AD HOC

# a = np.load('')
# np.savetxt('',a,delimiter=',',fmt='%d')

#==============================================================================#
#### read in peak count file and make distribution plot #### AD HOC

# myNet.peakCount = np.load('')
# from scipy.stats import skew,kurtosis
# print(max(myNet.peakCount), min(myNet.peakCount), np.median(myNet.peakCount),
#     np.mean(myNet.peakCount), skew(myNet.peakCount),kurtosis(myNet.peakCount))
# myNet.plotPeakCountDist('',histogram=False,standardize=False)
# myNet.plotPeakCountDist('',histogram=False,standardize=True)

#==============================================================================#
#### run time series for an FHN network with neuronal gij ####

# size = myNet.size
# sigma = 2
# dt = 5e-4 # step size
# T0 = 2e3 # time steps (noise-free network)
# T1 = 2e3 # time steps (noisy network)
# ic = np.random.uniform(-1,1,size) # initial conditions
# icY = np.random.uniform(-1,1,size)

# epsilon = 0.1
# alpha = 0.95

# noise-free network for steady states
# myNet.initDynamics_FHN(ic,icY,epsilon,alpha,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)

# generate time series
# myNet.initDynamics_FHN(ic,icY,epsilon,alpha,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.plotDynamics('')

#==============================================================================#
#### run time series for a logistic network with neuronal gij ####

# size = myNet.size
# sigma = 0.5
# dt = 5e-4 # step size
# T0 = 2e3 # time steps (noise-free network)
# T1 = 2e3 # time steps (noisy network)
# r0 = 10
# r = [r0]*size # coef of intrinsic dynamics
# ic = np.random.uniform(0.9,1.1,size) # initial conditions

# noise-free network for steady states
# myNet.initDynamics(ic,r,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)
# myNet.printDynamics('')
# myNet.plotDynamics('')
# myNet.setSteadyStates('')

# generate time series
# myNet.initDynamics(ic,r,sigma**2*np.eye(size))
# myNet.continueDynamics('',r,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.printDynamics('')
# myNet.plotDynamics('')
# myNet.removeTransient()

# info matrix
# myNet.calcTimeAvg()
# myNet.calcInfoMatrix()
# myNet.checkLogmCondition()
# myNet.estInfoMatrix()

# myNet.plotInfoMatrix('')

#==============================================================================#
#### run time series for a logistic network with Gaussian gij ####

# size = 100
# p = 0.2
# sigma = 1
# dt = 5e-4 # step size
# T0 = 2e3 # time steps (noise-free network)
# T1 = 2e3 # time steps (noisy network)
# w = [10,2] # weight
# r0 = 10
# r = [r0]*size # coef of intrinsic dynamics
# ic = np.random.uniform(0,5,size) # initial conditions

# myNet = n.network()
# myNet.randomDirectedWeightedGraph(size,p,w[0],w[1])

# noise-free network for steady states
# myNet.initDynamics(ic,r,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)
# myNet.printDynamics('')
# myNet.plotDynamics('')
# myNet.setSteadyStates()

# generate time series
# myNet.initDynamics(ic,r,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.printDynamics('')
# myNet.plotDynamics('')
# myNet.removeTransient()

# info matrix
# myNet.calcTimeAvg()
# myNet.calcInfoMatrix()
# myNet.checkLogmCondition()
# myNet.estInfoMatrix()

# myNet.plotInfoMatrix('')
# myNet.plotEstInfoMatrix('')
