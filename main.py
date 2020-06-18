import numpy as np
import matplotlib.pyplot as plt
import network_dynamics as n
np.random.seed(0)

size = 20
sigma = 1

myNet = n.network()
myNet.randomDirectedWeightedGraph(size,0.2,10,2)
myNet.printAdjacency('adj.txt')
myNet.printCoupling('cou.txt')
myNet.initDynamics(np.random.uniform(0,5,size),[10]*size,sigma**2*np.eye(size))
myNet.runDynamics(5e-4,2e3)
myNet.plotDynamics('dyn.png')
myNet.printDynamics('dyn.csv')
