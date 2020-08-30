# this file does the actual work
# useful code sections are cached below
import numpy as np
import matplotlib.pyplot as plt
import network_dynamics_cluster as n
# np.random.seed(0)

myNet = n.network()
# myNet.loadGraph('DIV25_PREmethod')

# a = np.load('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.01_a=0.95)PeakCount_t=0to500_h=0_d=2000.npy')
# b = np.loadtxt('DIV25_spks',usecols=0)
# idx = list(range(4095))

# fig = plt.figure()
# plt.scatter(a,np.log(b),s=1,c='k')
# plt.xlabel('model peak count')
# plt.ylabel('log(neuron peak count)')
# fig.tight_layout()
# fig.savefig('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.01_a=0.95)PeakCountvsNeurons.png')
# plt.close()

# fig = plt.figure()
# plt.scatter(idx,a,s=1,c='k')
# plt.xlabel('node index')
# plt.ylabel('model peak count')
# fig.tight_layout()
# fig.savefig('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=1.05)PeakCountvsIdx.png')
# plt.close()

# myNet.peakCount = np.load('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=1)PeakCount_t=0to500_h=0_d=2000.npy')
#
# from scipy.stats import skew,kurtosis
# print(max(myNet.peakCount), min(myNet.peakCount), np.median(myNet.peakCount),
#     np.mean(myNet.peakCount), skew(myNet.peakCount),kurtosis(myNet.peakCount))

# myNet.plotPeakCountAgainstDegStren('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=0.95)')
# myNet.plotPeakCountDist('(neuron_FHN_diffusive_withNoise_sigma=2_e=0.1_a=1.05)PeakCountDist_1e6steps.png',histogram=True,align='mid',bins=22)

#==============================================================================#

# size = 500
# size = myNet.size
#
# p = 0.2
# sigma = 1
# dt = 5e-4 # step size
# T0 = 2e2 # time steps (noise-free network)
# T1 = 2e4 # time steps (noisy network)
# w = [10,2] # weight
# ic = np.random.uniform(-1,1,size) # initial conditions
# icY = np.random.uniform(-1,1,size)

# myNet.randomDirectedWeightedGraph(size,p,w[0],w[1])

# epsilon = 0.001
# alpha = 0.95

# noise-free network for steady states
# myNet.initDynamics_FHN(ic,icY,epsilon,alpha,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)

# myNet.initDynamics_FHN(ic,icY,epsilon,alpha,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.plotDynamics('myPlot.png')

#==============================================================================#

# myNet.loadNpGraph('DIV25_PREmethod_GaussianRef.npy')
# myNet.plotStrengthDist('(neuron)')

# from scipy.stats import gaussian_kde,norm
# fig = plt.figure()
# a = myNet.Coupling[myNet.Coupling>0]
# b = abs(myNet.Coupling[myNet.Coupling<0])
# c = np.concatenate([a,b])
# cmin = np.percentile(c,0)
# cmax = np.percentile(c,99)
# x = np.linspace(cmin,cmax,200)
# density_a = gaussian_kde(a)
# density_b = gaussian_kde(b)
# plt.plot(x,density_a(x),'r',label='$g_{ij}>0$')
# plt.plot(x,density_b(x),'b',label='$g_{ij}<0$')
# plt.plot(x,norm(loc=a.mean(),scale=a.std()).pdf(x),'r--')
# plt.plot(x,norm(loc=b.mean(),scale=b.std()).pdf(x),'b--')
# plt.ylim(bottom=0)
# plt.xlabel('$|g_{ij}|$')
# plt.legend()
# fig.tight_layout()
# fig.savefig('myPlot1.png')
# plt.close()
#
# print(np.mean(a),np.median(a))
# print(np.mean(b),np.median(b))
#
# myNet.loadNpGraph('DIV25_PREmethod_GaussianRef.npy')
# fig = plt.figure()
# _a = myNet.Coupling[myNet.Coupling>0]
# _b = abs(myNet.Coupling[myNet.Coupling<0])
# c = np.concatenate([_a,_b])
# cmin = np.percentile(c,0)
# cmax = np.percentile(c,99)
# x = np.linspace(cmin,cmax,200)
# density_a = gaussian_kde(_a)
# density_b = gaussian_kde(_b)
# plt.plot(x,density_a(x),'r',label='$g_{ij}>0$')
# plt.plot(x,density_b(x),'b',label='$g_{ij}<0$')
# plt.plot(x,norm(loc=a.mean(),scale=a.std()).pdf(x),'r--')
# plt.plot(x,norm(loc=b.mean(),scale=b.std()).pdf(x),'b--')
# plt.ylim(bottom=0)
# plt.xlabel('$|g_{ij}|$')
# plt.legend()
# fig.tight_layout()
# fig.savefig('myPlot2.png')
# plt.close()
#
# print(np.mean(_a),np.median(_a))
# print(np.mean(_b),np.median(_b))

#==============================================================================#

# myNet.loadGraph('DIV25_PREmethod',multiplier=10)
# myNet.setAsGaussianRefGraph()
# myNet.plotStrengthDist('(neuronGaussianRef)')
# myNet.plotDegreeStrengthDist('myDeg.png','myStren.png')

# myNet.peakCount = np.load('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100_ref)PeakCount_t=0to50_h=1+2sd_d=20.npy')
# myNet.plotPeakCountAgainstDegStren('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100_ref)')
# myNet.plotPeakCountDist('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100_ref)PeakCountDist.png')

# a = np.load('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100_ref)PeakCount_t=0to50_h=1+2sd_d=20.npy')
# b = np.load('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100)PeakCount_t=0to50_h=1+2sd_d=20.npy')
#
# fig = plt.figure()
# plt.scatter(a,b,s=1,c='k')
# plt.xlabel('reference model peak count')
# plt.ylabel('model peak count')
# fig.tight_layout()
# fig.savefig('(neuron_mult10_diffusive_withNoise_sigma=1.4_r0=100)PeakvsRef.png')
# plt.close()

# myNet.readDynamics(['(neuron_noiseFree)dyn_dt=2e-4_5e3steps.npy','(neuron_noiseFree)dyn_dt=2e-4_5e3to6e3steps.npy'])
# myNet.readDynamics('(neuron_withNoise_sigma=0.5_y0=4)dyn_dt=2e-4_2e4steps.npy')
# myNet.saveNpDynamics('(neuron_withNoise_sigma=0.5_y0=4)dyn_dt=2e-4_2e4steps.npy')
# myNet.setSteadyStates('(neuron_noiseFree)dyn_lastStep.csv')
# myNet.removeTransient(4000)
# myNet.calcStatesFluc()
# myNet.plotFlucDist('flucDist.png',[0,4])
# myNet.plotFlucSdAgainstDegStren('SdDeg.png','SdStren.png')

# size = myNet.size
# sigma = 0.5
# dt = 2e-4 # step size
# T0 = 5e2 # time steps (noise-free network)
# T1 = 6e3 # time steps (noisy network)
# r0 = 10
# r = [r0]*size # coef of intrinsic dynamics
# ic = np.random.uniform(0.9,1.1,size) # initial conditions
# c = 'synaptic' # coupling function

# myEmail = 'frankieycy@gmail.com'
# myPw = 'ycy19990414'
# myNet.setEmail(myEmail,myPw,myEmail)

# noise-free network for steady states
# myNet.emailHandler.setEmailTitle('noise-free network')
# myNet.initDynamics(ic,r,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)
# myNet.printDynamics('dyn.csv')
# myNet.plotDynamics('dyn.png')
# myNet.setSteadyStates('(neuron_noiseFree)dyn_lastStep.csv')

# generate time series
# myNet.emailHandler.setEmailTitle('noisy network')
# myNet.initDynamics(ic,r,sigma**2*np.eye(size))
# myNet.continueDynamics('(cont)(neuron_noiseFree)dyn_dt=2e-4_5e3steps.csv',r,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.printDynamics('(neuron_noiseFree)dyn_dt=2e-4_5e3to6e3steps.csv')
# myNet.plotDynamics('(neuron_withNoise_sigma=0.5_y0=4_)dyn_dt=2e-4_5e3steps.png')
# myNet.removeTransient(4000)

# info matrix
# myNet.calcTimeAvg()
# myNet.calcInfoMatrix()
# myNet.checkLogmCondition()
# myNet.estInfoMatrix()

# t = 'size $=%d$, effective weight, $r_i = %d$, %s coupling'%(size,r0,c)
# myNet.plotInfoMatrix('QvsM.png',t)

# myNet.emailHandler.quitEmail()

#==============================================================================#

# size = 100
# p = 0.2
# sigma = 1
# dt = 5e-4 # step size
# T0 = 2e3 # time steps (noise-free network)
# T1 = 2e4 # time steps (noisy network)
# w = [10,2] # weight
# r0 = 10
# r = [r0]*size # coef of intrinsic dynamics
# ic = np.random.uniform(0,5,size) # initial conditions
# c = 'synaptic' # coupling function

# myNet = n.network()
# myNet.randomDirectedWeightedGraph(size,p,w[0],w[1])

# noise-free network for steady states
# myNet.initDynamics(ic,r,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)
# myNet.printDynamics('dyn.csv')
# myNet.plotDynamics('dyn.png')
# myNet.setSteadyStates()

# generate time series
# myNet.initDynamics(ic,r,sigma**2*np.eye(size))
# myNet.runDynamics(dt,T1)
# myNet.printDynamics('dyn.csv')
# myNet.plotDynamics('dyn.png')
# myNet.removeTransient(400)

# info matrix
# myNet.calcTimeAvg()
# myNet.calcInfoMatrix()
# myNet.checkLogmCondition()
# myNet.estInfoMatrix()

# t = 'size $=%d$, weight $\sim N(%d,%d)$, $r_i = %d$, %s coupling'%(size,w[0],w[1],r0,c)
# myNet.plotInfoMatrix('QvsM.png',t)
# myNet.plotEstInfoMatrix('info.png',t)
