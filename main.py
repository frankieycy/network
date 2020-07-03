import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
# np.random.seed(0)

myNet = n.network()
myNet.loadGraph('neuronCou.txt')

size = myNet.size
# size = 100
# p = 0.2
sigma = 1
dt = 5e-4 # step size
T = 2e4 # time steps
# w = [10,2] # weight
r0 = 10
r = [r0]*size # coef of intrinsic dynamics
ic = np.random.uniform(0.9,1.1,size) # initial conditions
# c = 'synaptic' # coupling function

# myNet = n.network()
# myNet.randomDirectedWeightedGraph(size,p,w[0],w[1])

# noise-free network for steady states
myNet.initDynamics(ic,r,np.zeros((size,size)))
myNet.runDynamics(dt,4e3)
myNet.setSteadyStates()

# generate time series
myNet.initDynamics(ic,r,sigma**2*np.eye(size))
myNet.runDynamics(dt,T)
# myNet.plotDynamics('dyn.png')
myNet.removeTransient(800)

# info matrix
myNet.calcTimeAvg()
myNet.calcInfoMatrix()
myNet.estInfoMatrix()

t = 'size $=%d$, weight $\sim N(%d,%d)$, $r_i = %d$, %s coupling'%(size,w[0],w[1],r0,c)
myNet.plotInfoMatrix('QvsM.png',t)
myNet.plotEstInfoMatrix('info.png',t)
