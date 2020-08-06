## Neuronal Network Analysis

* last: 7/8/2020
* reconstruct network from dynamics (time series data) of nodes
* replicate neuron dynamics with an effective network model (with connectivity computed from an assumed model)

## Project 1 - toy model

* paper: Extracting connectivity from dynamics of networks with uniform bidirectional coupling (2013)
* recover network connectivity (adjacency matrix) from time series data of a known random network
* nodes obey consensus dynamics
* code:
    - ``network.py``

## Project 2a - (FYP Prelim) simulated dynamics, connection extraction

* paper: Reconstructing links in directed networks from noisy dynamics (2017)
* simulate dynamics based on eq[1] - dynamics governed by intrinsic dynamics, node interaction, noise
* **intrinsic dynamics** (dynamics on its own): eq[10] - stable at 1, param r_i=r0
* **coupling function** (interaction between nodes): eq[12] - synaptic/diffusive, param (beta1,beta2,y0)
* **Gaussian white noise** (iid.): param sigma

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

* (neuron, synaptic coupling function) parameters (r0,beta1,beta2,y0,sigma) that lead to stable (non-diverging) time series:
    - (100,2,0.5,1,0.5)
    - (10,20,1,1,0.25)
    - this model is not very successful to replicate neuron dynamics

* (neuron, diffusive coupling function) parameters (r0,g_ij multiplier,sigma) that lead to stable (non-diverging) time series:
    - (10,10,0.25)
    - (100,10,1.5)
    - this model is not very successful to replicate neuron dynamics

* **new research directions**:
    - does model dynamics resemble (experimental) neuron dynamics?
    - how do spiking activities of model dynamics vary with different network features (e.g. degrees/strengths)?
    - does heavy-tailed distribution in coupling strengths explain heavy-tailed spiking activities?

* code:
    - ``network_dynamics_cluster.py``: for use in physics dept clusters
