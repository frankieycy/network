import numpy as np
import matplotlib.pyplot as plt
import network_dynamics_torch as n
np.random.seed(0)

myNet = n.network()
myNet.loadGraph('DIV25_PREmethod')
# myNet.readDynamics('dyn.csv')
# myNet.plotDynamics('dyn.png')

size = myNet.size
sigma = 0.5
dt = 2e-4 # step size
# T0 = 5e3 # time steps (noise-free network)
T1 = 5e3 # time steps (noisy network)
r0 = 10
r = [r0]*size # coef of intrinsic dynamics
ic = np.random.uniform(0.9,1.1,size) # initial conditions
c = 'synaptic' # coupling function

# myEmail = 'frankieycy@gmail.com'
# myPw = 'ycy19990414'
# myNet.setEmail(myEmail,myPw,myEmail)

# noise-free network for steady states
# myNet.emailHandler.setEmailTitle('noise-free network')
# myNet.initDynamics(ic,r,np.zeros((size,size)))
# myNet.runDynamics(dt,T0)
# myNet.printDynamics('dyn.csv')
# myNet.plotDynamics('dyn.png')
myNet.setSteadyStates('(neuron_noiseFree)dyn_lastStep.csv')

# generate time series
# myNet.emailHandler.setEmailTitle('noisy network')
myNet.initDynamics(ic,r,sigma**2*np.eye(size))
# myNet.continueDynamics('dyn.csv',r,sigma**2*np.eye(size))
myNet.runDynamics(dt,T1)
myNet.printDynamics('(neuron_withNoise_sigma=0.5_y0=4_)dyn_dt=2e-4_5e3steps.csv')
myNet.plotDynamics('(neuron_withNoise_sigma=0.5_y0=4_)dyn_dt=2e-4_5e3steps.png')
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
