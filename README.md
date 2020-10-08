## Neuronal Network Analysis

* last update: 8/10/2020
* _the README just documents my work and is not meant to be self-contained_
* reconstruct network from dynamics (time series data) of nodes
* replicate neuron dynamics with an effective network model (with connectivity computed from an assumed model)

## Project 1 - toy model

* paper: Extracting connectivity from dynamics of networks with uniform bidirectional coupling (2013)
* recover network connectivity (adjacency matrix) from time series data of a known random network
* nodes obey consensus dynamics
* code:
    - ``network.py``

## Project 2 - (FYP) synaptic/diffusive neuronal network

* paper: Reconstructing links in directed networks from noisy dynamics (2017)
* simulate dynamics based on eq[1] - dynamics governed by intrinsic dynamics, node interaction, noise
* **intrinsic dynamics** (dynamics on its own): eq[10] - _logistic_, stable point at 1, param r_i = r0
* **coupling function** (interaction between nodes): eq[12] - _synaptic/diffusive_, param (beta1, beta2, y0)
* **Gaussian white noise** (iid.): param sigma

---

* (case 4) default parameters:
    - random directed weighted graph
    - size = 100, connection probability = 0.2, weights ~ N(10,2), noise sigma = 1
    - intrinsic dynamics coefficient: r_i = 10
    - synaptic coupling function: (beta1, beta2, y0) = (2, 0.5, 4)
    - initial conditions: uniform[0,5]
    - step size = 5e-4, time steps = 2e6
    - subject to modifications for different test cases

---

* [1] **question: replicate neuronal dynamics with model-estimated links and weights**
    - from time series data, links and weights were estimated from an assumed network model
    - can the network constructed from those links and weights recover features in the time series data?

* experimental observation of neurons:
    - non-Gaussian, heavy-tailed spiking
    - a majority of nodes have zero to one spike while some nodes have far more spikes (a few orders greater) than others

* (synaptic coupling function) parameters (r0, beta1, beta2, y0, sigma) that lead to stable (non-diverging) time series:
    - (100, 2, 0.5, 1, 0.5)
    - (10, 20, 1, 1, 0.25)
    - _this model is not very successful in replicating neuron dynamics_

* (diffusive coupling function) parameters (r0, g_ij multiplier, sigma) that lead to stable (non-diverging) time series:
    - (10, 10, 0.25)
    - (100, 10, 1.5)
    - _this model is not very successful in replicating neuron dynamics_

* **findings**:
    - network connectivity does not fully explain neuron spiking activities
    - either neuron dynamics may not fit into the network framework (this runs counter to our intuition), or
    - the model oversimplifies neuron dynamics (i.e. may require other elements like periodic firing)

---

* [2] **new research directions**:
    - does model dynamics resemble real neuron dynamics? how do their distributions of spikes differ?
    - how do spiking activities of model dynamics vary with different network features (e.g. degrees/strengths)? here we are concerned only with the model itself though it is unrealistic
    - does heavy-tailed distribution in coupling strengths explain heavy-tailed spiking activities?

## Project 3 - (FYP) FHN neuronal network

* same dynamical equation - eq[1], but with the logistic intrinsic dynamics replaced by the FHN one
* this is a 2D model, i.e. each node has (x_i, y_i) states

* (diffusive coupling function) parameters (epsilon, alpha, sigma) that lead to spiking dynamics:
    - (0.01, 0.95, 2): both noise-free and with-noise time series have spikes
    - (0.1, 0.95, 2): both noise-free and with-noise time series have spikes (this set most resembles real neuron dynamics)
    - (0.1, 1, 2): the noise-free time series have decaying oscillations, but the with-noise time series have spikes
    - (0.1, 1.05, 2): the noise-free time series have no spiking/oscillatory activities, but the with-noise time series have spikes

* This model is GREAT!
    - exhibits realistic spiking patterns
    - recovers heavy-tailed nature of spike tail
    - rich correlations between spike counts and network features
    - see report for full analysis

* code: (for both project 2 & 3)
    - ``network_dynamics_cluster.py``: for use in physics department clusters

---

* This marks the end of my FYP part I.
