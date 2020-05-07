import numpy as np
import matplotlib.pyplot as plt
import network as n
np.random.seed(0)

s = 20 # network size
g = n.network(size=s,connectProb=.2,noiseSigma=1) # generate random network
g.printAdjacency('adj.txt') # print adjacency matrix to file
g.initDynamics(np.random.uniform(0,1,s)) # random initial conditions
g.runDynamics(5e-4,1000) # iterate dynamics
g.estimateConnectivity() # estimate connectivity from time series data
