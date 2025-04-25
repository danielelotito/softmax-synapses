# Memory Evolution with Softmax-Weighted Synaptic Updates

This repository contains the numerical implementation accompanying Appendix B of the paper:

**Neural Learning Rules from Associative Networks Theory**  
*Daniele Lotito (2025)*  
Published on Neurocomputing
(https://doi.org/10.1016/j.neucom.2025.129865)

## Overview

This code demonstrates the evolution of memory matrices through softmax-weighted synaptic updates, comparing numerical simulations with theoretical predictions. The implementation focuses on verifying the convergence of memory dynamics to Hebbian learning prescriptions in neural networks.

## Theoretical Background

The core of this work investigates memory dynamics governed by the differential equation:

```
τ_Ξ Ξ̇ = w ⊙ [(-Ξ + Λ) + v^T]
```

where:
- Ξ represents the memory matrix
- w is a weight vector computed through a softmax function
- Λ is the synaptic bias matrix
- v is the visible state vector
- ⊙ denotes the Hadamard (element-wise) product

The weight vector w is determined by:

```
w_μ = exp(β(Ξv)_μ) / Σ_ν exp(β(Ξv)_ν)
```

where β controls the selectivity of the weighting.

## Implementation Details

The code implements:
1. Creation of orthogonal memory matrices using Gram-Schmidt process
2. Evolution of memory dynamics with softmax-weighted updates
3. Comparison between numerical results and theoretical predictions
4. Visualization of memory similarity evolution over time

Key parameters:
- N_v: Number of visible neurons
- N_h: Number of hidden neurons
- T: Total simulation time
- τ_Ξ: Memory time constant
- β: Softmax temperature parameter

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Usage

Run the main script:

```bash
python memory-dynamics-theoretical.py
```

This will generate a plot comparing the numerical evolution of memory similarities with theoretical predictions.

## Author

Daniele Lotito  
Contact: name.surname1@gmail.com

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lotito2025neural,
title = {Neural learning rules from associative networks theory},
journal = {Neurocomputing},
volume = {636},
pages = {129865},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.129865},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225005375},
author = {Daniele Lotito}
}
```
