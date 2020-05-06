import numpy as np
import matplotlib.pyplot as plt
import network as n

# g = n.graph(15,.2,1)
# g.printAdjacency('adj.txt')
# g.printCoupling('cou.txt')

# g = n.graph(adjacencyFile='adj.txt')
# print(g.getLaplacian())

s = 100
g = n.network(size=s,connectProb=.2,noiseSigma=1)
g.printAdjacency('adj.txt')
g.initDynamics(np.random.uniform(0,1,s))
g.runDynamics(5e-4,50)
g.estimateConnectivity()
