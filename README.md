# tensor-mdp-solver
MATLAB &amp; Python implementations of an exact, tensor-based MDP solver for life-cycle optimization of civil infrastructure

# Tensor-Based MDP Solver for Civil Infrastructure Systems

[![arXiv](https://img.shields.io/badge/arXiv-2604.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides MATLAB and Python implementations of an exact, tensor-based Markov Decision Process (MDP) solver. 

Standard dynamic programming solvers suffer from the **curse of dimensionality**, where the state and action spaces grow exponentially with the number of components. This solver leverages the Kronecker product decomposition to evaluate the Bellman optimality update without ever constructing the full, intractable transition matrix. It scales linearly rather than exponentially, allowing for exact global optimal policies on large-scale deteriorating civil infrastructure networks.

## Full Paper
This code is the companion to our paper:
> **"Probabilistic Hazard Analysis Framework with Stochastic Optimal Control for Deteriorating Civil Infrastructure Systems"** > Read the full preprint on arXiv: [https://arxiv.org/abs/2604.23068]

##  Key Features
**Exact Solutions:** Computes the true optimal policy without approximation techniques.
**Memory Efficient:** Avoids storing the massive joint transition matrix by utilizing sequential mode-$k$ tensor products.
**Dual Implementations:** Full source code available in both MATLAB and Python.

## Setup

### Python
1. Clone the repository:
   ```bash
   git clone [https://github.com/jodha-sudhir/tensor-mdp-solver.git](https://github.com/jodha-sudhir/tensor-mdp-solver.git)
   cd tensor-mdp-solver/python
