# Improved Kernel PLS
Fast CPU, GPU, and TPU Python implementations of Improved Kernel PLS Algorithm #1 and Algorithm #2 by Dayal and MacGregor[^1]. Improved Kernel PLS has been shown to be both fast[^2] and numerically stable[^3].
The CPU implementations are made using NumPy[^4] and subclass BaseEstimator from scikit-learn[^5] allowing integration into scikit-learn's ecosystem of machine learning algorithms and pipelines. For example, the CPU implementations can be used with scikit-learn's `cross_validate`.
The GPU and TPU implementations are made using Google's JAX[^6]. While allowing CPU, GPU, and TPU execution, automatic differentiation is also supported by JAX. This implies that the JAX implementations can be used together with deep learning approaches as the PLS fit is differentiable.

[^1]: [Dayal, B. S., & MacGregor, J. F. (1997). Improved PLS algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23)
[^2]: [Alin, A. (2009). Comparison of PLS algorithms when number of objects is much larger than number of variables. Statistical papers, 50, 711-720.](https://link.springer.com/content/pdf/10.1007/s00362-009-0251-7.pdf)
[^3]: [Andersson, M. (2009). A comparison of nine PLS1 algorithms. Journal of Chemometrics: A Journal of the Chemometrics Society, 23(10), 518-529.](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.1248?)
[^4]: [NumPy.](https://numpy.org/)
[^5]: [scikit-learn.](https://scikit-learn.org/stable/)
[^6]: [JAX.](https://jax.readthedocs.io/en/latest/)


## Pre-requisites
The JAX implementations support running on both CPU, GPU, and TPU. To use the GPU or TPU, follow the instructions from the [JAX Installation Guide](https://jax.readthedocs.io/en/latest/installation.html).
To ensure that JAX implementations use Float64, set the environment variable JAX_ENABLE_X64=True as per the [Current Gotchas](https://github.com/google/jax#current-gotchas).

## Installation
* Install the package for Python3 using the following command:
`$ pip install ikpls`
* Now you can import the NumPy and JAX implementations with:
```python
from ikpls.numpy_ikpls import PLS as NpPLS
from ikpls.jax_ikpls_alg_1 import PLS as JAXPLS_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAXPLS_Alg_2
```

## Quick Start
### Use the ikpls package for PLS modelling
```python
from ikpls.numpy_ikpls import PLS as NpPLS
import numpy as np

N = 100 # Number of samples
K = 50 # Number of targets
M = 10 # Number of features
A = 20 # Number of latent variables (PLS components)

X = np.random.uniform(size=(N, K), dtype=np.float64)
Y = np.random.uniform(size=(N, M), dtype=np.float64)

# The other PLS algorithms and implementations have the same interface for fit() and predict().
nppls_alg_1 = NpPLS()
nppls_alg_1.fit(X, Y, A)

y_pred = nppls_alg_1.predict(X) # Will have shape (A, N, M)
y_pred_20_components = nppls_alg_1.predict(X, n_components=20) # Will have shape (N, M)
```

## Examples
In [examples](examples/) you will find:
* [Example of fitting and predicting with the NumPy implementations.](examples/fit_predict_numpy.py)
* [Example of fitting and predicting with the JAX implementations.](examples/fit_predict_jax.py)
* [Example of cross validating with the NumPy implementations.](examples/cross_val_numpy.py)
* [Example of cross validating with the JAX implementations.](examples/cross_val_jax.py)
* [Example of computing the gradient of a preprocessing filter with respect to the RMSE between the target value and the value predicted by PLS after fitting.](examples/gradient_jax.py)