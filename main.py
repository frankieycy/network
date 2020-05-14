import numpy as np
import matplotlib.pyplot as plt
import network as n
np.random.seed(0)

s = 20 # network size
g = n.network()
g.randomUniformGraph(s,connectProb=.2) # generate random network
g.initDynamics_Concensus(np.random.uniform(0,1,s),noiseSigma=1) # random initial conditions
g.printAdjacency('adj.txt') # print adjacency matrix to file
g.runDynamics(5e-4,100) # iterate dynamics
g.estimateConnectivity() # estimate connectivity from time series data
g.showAnalysis()
