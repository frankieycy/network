import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
# np.random.seed(0)

myNet = n.network()
myNet.loadGraph('neuronCou.txt')

size = myNet.size
# size = 1000
# p = 0.2
sigma = 0.5
dt = 2e-4 # step size
T0 = 5e3 # time steps (noise-free network)
T1 = 5e4 # time steps (noisy network)
# w = [10,2] # weight
r0 = 10
r = [r0]*size # coef of intrinsic dynamics
ic = np.random.uniform(0.9,1.1,size) # initial conditions
c = 'synaptic' # coupling function

# myNet = n.network()
# myNet.randomDirectedWeightedGraph(size,p,w[0],w[1])

# myEmail = 'frankieycy@gmail.com'
# myPw = 'ycy19990414'
# myNet.setEmail(myEmail,myPw,myEmail)

# noise-free network for steady states
# myNet.emailHandler.setEmailTitle('noise-free network')
myNet.initDynamics(ic,r,np.zeros((size,size)))
myNet.runDynamics(dt,T0)
# myNet.plotDynamics('dyn-a.png')
myNet.setSteadyStates()

# generate time series
# myNet.emailHandler.setEmailTitle('noisy network')
myNet.initDynamics(ic,r,sigma**2*np.eye(size))
myNet.runDynamics(dt,T1)
# myNet.plotDynamics('dyn-b.png')
myNet.removeTransient(4000)

# info matrix
myNet.calcTimeAvg()
myNet.calcInfoMatrix()
myNet.checkLogmCondition()
myNet.estInfoMatrix()

# t = 'size $=%d$, weight $\sim N(%d,%d)$, $r_i = %d$, %s coupling'%(size,w[0],w[1],r0,c)
t = 'size $=%d$, effective weight, $r_i = %d$, %s coupling'%(size,r0,c)
myNet.plotInfoMatrix('QvsM.png',t)
# myNet.plotEstInfoMatrix('info.png',t)

# myNet.emailHandler.quitEmail()
