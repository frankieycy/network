import numpy as np
import matplotlib.pyplot as plt
import network as n

# g = n.graph(15,.2,1)
# g.printAdjacency('adj.txt')
# g.printCoupling('cou.txt')

# g = n.graph(15,adjacencyFile='adj.txt',couplingFile='cou.txt')
# print(g.getLaplacian())

g = n.network(15,intrinsicFunc=n.f,couplingFunc=n.h,noiseSigma=1,\
    adjacencyFile='adj.txt',couplingFile='cou.txt')
