## Neuronal Network Analysis

* last: 13/7/2020
* reconstruct network from dynamics (time series data) of nodes
* replicate neuron dynamics with an effective network model

## Project 1 - toy model

* paper: Extracting connectivity from dynamics of networks with uniform bidirectional coupling (2013)
* recover network connectivity (adjacency matrix) from time series data of a known random network
* nodes obey consensus dynamics
* code:
    - ``network.py``

## Project 2a - (FYP Prelim) simulated dynamics, connection extraction

* paper: Reconstructing links in directed networks from noisy dynamics (2017)
* simulate dynamics based on eq[1] - dynamics governed by intrinsic dynamics, node interaction, noise
* intrinsic dynamics (dynamics on its own): eq[10] - stable at 1, param r_i=r0
* coupling function (interaction between nodes): eq[12] - synaptic, param (beta1,beta2,y0)
* Gaussian white noise (iid.): param sigma

* (case 4) default parameters:
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
* **goal: replicate neuronal dynamics with model-estimated links and weights**
* (neuron) parameters (r0,beta1,beta2,y0,sigma) that lead to stable (non-diverging) time series:
    - (100,2,0.5,1,0.5)
    - (10,20,1,1,0.25)
    - _more later_

* code:
    - ``network_dynamics_cluster.py``: for use in physics dept clusters
