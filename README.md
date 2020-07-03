## Neuronal Network Analysis

* last: 3/7/2020
* reconstruct network from dynamics (time series data) of nodes
* replicate neuronal dynamics from model

## Project 1 - toy model

* paper: Extracting connectivity from dynamics of networks with uniform bidirectional coupling (2013)
* recover network connectivity (adjacency matrix) from time series data of a known random network
* nodes obey consensus dynamics
* code: ``network.py``

## Project 2a - (FYP Prelim) simulated dynamics, connection extraction

* paper: Reconstructing links in directed networks from noisy dynamics (2017)
* simulate dynamics based on eq[1] - dynamics governed by intrinsic dynamics, node interaction, noise
* intrinsic dynamics (dynamics on its own): eq[10] - stable at 1
* coupling function (interaction between nodes): eq[12] - synaptic
* parameters (see ``main.py``):
    - random directed weighted graph
    - size = 100, connection probability = 0.2, weights ~ N(10,2), noise sigma = 1
    - intrinsic dynamics coefficient: r_i = 10
    - synaptic coupling function: beta1,beta2,y0 = 2,0.5,4
    - initial conditions: uniform[0,5]
    - step size = 5e-4, time steps = 2e6
    - subject to modifications for different test cases
* experimental observation of neurons:
    - non-uniform spiking
    - some larger activity, some smaller
* *goal: replicate neuronal dynamics with model-estimated links and weights*
* code: ``network_dynamics.py``
