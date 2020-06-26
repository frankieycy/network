import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
np.random.seed(0)

size = 20
sigma = 1

myNet = n.network()
myNet.randomDirectedWeightedGraph(size,0.2,10,2)

# noise-free network for steady states
myNet.initDynamics(np.random.uniform(0,5,size),[10]*size,np.zeros((size,size)))
myNet.runDynamics(5e-4,2e3)
myNet.setSteadyStates()

myNet.initDynamics(np.random.uniform(0,5,size),[10]*size,sigma**2*np.eye(size))
myNet.runDynamics(5e-4,2e4)
# myNet.plotDynamics('dyn.png')
myNet.removeTransient(400)
myNet.calcTimeAvg()
myNet.calcInfoMatrix()
myNet.plotInfoMatrix('QvsM.png')

################################################################################
# myNet.printAdjacency('adj.txt')
# myNet.printCoupling('cou.txt')
# myNet.initDynamics(np.random.uniform(0,5,size),[10]*size,sigma**2*np.eye(size))
# myNet.runDynamics(5e-4,2e3)
# myNet.plotDynamics('dyn.png')
# myNet.printDynamics('dyn.csv')
