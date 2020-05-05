import numpy as np
import matplotlib.pyplot as plt
import network as n

def f(x):
    return 0

def h(xi,xj):
    return xi-xj

# g = n.graph(15,.2,1)
# g.printAdjacency('adj.txt')
# g.printCoupling('cou.txt')

# g = n.graph(15,adjacencyFile='adj.txt',couplingFile='cou.txt')
# print(g.getLaplacian())

s = 100
g = n.network(s,intrinsicFunc=f,couplingFunc=h,noiseSigma=.1,coupling=1,connectProb=.05)
g.printAdjacency('adj.txt')
g.initDynamics(np.random.rand(s))
g.runDynamics(.01,10)
g.printDynamics('data.csv')
g.plotDynamics('data.png')
