import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
# np.random.seed(0)

size = 100
sigma = 1
dt = 5e-4 # step size
T = 2e4 # time steps
w = [10,2] # weight
r = [50]*size # coef of intrinsic dynamics
ic = np.random.uniform(0,5,size) # initial conditions
c = 'synaptic' # coupling function

myNet = n.network()
myNet.randomUnidirectedWeightedGraph(size,0.2,w[0],w[1])

# noise-free network for steady states
myNet.initDynamics(ic,r,np.zeros((size,size)))
myNet.runDynamics(dt,2e3)
myNet.setSteadyStates()

# generate time series
myNet.initDynamics(ic,r,sigma**2*np.eye(size))
myNet.runDynamics(dt,T)
myNet.removeTransient(400)
# myNet.plotDynamics('dyn.png')

# info matrix
myNet.calcTimeAvg()
myNet.calcInfoMatrix()
myNet.estInfoMatrix()

t = 'size $=%d$, weight $\sim N(%d,%d)$, $r_i = %d$, %s coupling'%(size,w[0],w[1],r[0],c)
myNet.plotInfoMatrix('QvsM.png',t)
myNet.plotEstInfoMatrix('info.png',t)
