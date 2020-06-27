import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
# np.random.seed(0)

size = 100
sigma = 1
dt = 5e-4
w = [10,2]
r = [50]*size
ic = np.random.uniform(0,5,size)

myNet = n.network()
myNet.randomUnidirectedWeightedGraph(size,0.2,w[0],w[1])

# noise-free network for steady states
myNet.initDynamics(ic,r,np.zeros((size,size)))
myNet.runDynamics(dt,2e3)
myNet.setSteadyStates()

myNet.initDynamics(ic,r,sigma**2*np.eye(size))
myNet.runDynamics(dt,2e5)
myNet.removeTransient(400)
# myNet.plotDynamics('dyn.png')

myNet.calcTimeAvg()
myNet.calcInfoMatrix()
myNet.estInfoMatrix()
myNet.plotInfoMatrix('QvsM.png','size $=%d$, weight $\sim N(%d,%d)$, $r_i = %d$, %s coupling'\
    %(size,w[0],w[1],r[0],'diffusive'))
