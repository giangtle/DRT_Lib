# Distribution of Relaxation Times Library (DRT_Lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of different algorithms for calculating electrochemical impedance distribution of relaxation times. 
## Installation
```
pip install -i https://test.pypi.org/simple/ DRT-Lib==0.0.4
```
or
```
npm install @giangtle/drt_lib@1.0.1
```
or
```
"@giangtle/drt_lib": "1.0.1"
```
## Code Credits
Major credit to Liu and Ciucci source codes ([GitHub Page](https://github.com/ciuccislab), [Deep-Prior DRT](https://github.com/ciuccislab/DP-DRT), [Gaussian Process DRT](https://github.com/ciuccislab/GP-DRT)). Two algorithms for calculating DRT are based straight of their works & open-source codes.

## Methods Included
* Deep-Prior Distribution of Relaxation Times ([Liu and Ciucci, 2020](https://iopscience.iop.org/article/10.1149/1945-7111/ab631a))
* Gaussian Process Distribution of Relaxation Times ([Liu and Ciucci, 2020](https://www.sciencedirect.com/science/article/pii/S0013468619321887))
* Tikhonov Regularization/Ridge Regression Distribution of Relaxation Times ([Saccoccio et al., 2014](https://www.sciencedirect.com/science/article/pii/S0013468614018763))

## Quick Start
Checkout Example.ipynb file with jupyter notebook for more details.
```python
import DRT_Lib
import numpy as np
from math import pi

# Frequency range:
N_freqs = 81
freq_vec = np.logspace(-4, 4, N_freqs)

# Create sample electrochemical impedance spectra (EIS) of a ZARC element with noise:
# Parameters:
R_inf = 10
R_ct = 50
phi = 0.8
tau_0 = 1.
C = tau_0**phi/R_ct
# Exact impedance:
Z_exact = R_inf + 1./( 1./R_ct + C*(1j*2.*pi*freq_vec)**phi )
# Add noise to make synthetic experiment impedance data:
rng = np.random.seed(214975)
sigma_n_exp = 0.1
Z_exp = Z_exact + sigma_n_exp*( np.random.normal(0, 1, N_freqs) + 1j*np.random.normal(0, 1, N_freqs) )

# Compute Tikhonov Regularization/Ridge Regression Distrubution of Relaxation Times:
gamma, R_inf = DRT_Lib.TR_DRT(freq_vec, Z_exp, display=True)

# To caclulate EIS from DRT results:
Z_cal = DRT_Lib.calculate_EIS(freq_vec, gamma, R_inf)
```
## License
Released under the [MIT License](LICENSE).
