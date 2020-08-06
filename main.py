import numpy as np
import matplotlib.pyplot as plt
import network_dynamics_cluster as n
# np.random.seed(0)

myNet = n.network()
# myNet.loadGraph('DIV25_PREmethod',multiplier=10)

a = np.load('(neuron_mult10_diffusive_withNoise_sigma=1.5_r0=100)PeakCount_t=0to50_h=1+2sd_d=20.npy')
b = np.loadtxt('DIV25_spks',usecols=0)

m = a[b>0]
n = np.log(b[b>0])

fig = plt.figure()
plt.scatter(m,n,s=1,c='k')
plt.xlabel('model peak count')
plt.ylabel('log(neuron peak count)')
fig.tight_layout()
fig.savefig('(neuron_mult10_diffusive_withNoise_sigma=1.5_r0=100)PeakCountvsNueronLog.png')
plt.close()

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
