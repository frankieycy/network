import numpy as np
import matplotlib.pyplot as plt
import network as n
np.random.seed(0)

s = 20
g = n.network(size=s,connectProb=.2,noiseSigma=1)
g.printAdjacency('adj.txt')
g.initDynamics(np.random.uniform(0,1,s))
g.runDynamics(5e-4,1000)
g.estimateConnectivity()
