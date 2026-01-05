# Improved Kernel Partial Least Squares (IKPLS) and Fast Cross-Validation

[![PyPI Version](https://img.shields.io/pypi/v/ikpls)](https://pypi.python.org/pypi/ikpls/)

[![Python Versions](https://img.shields.io/pypi/pyversions/ikpls)](https://pypi.python.org/pypi/ikpls/)

[![Pepy - Total Downloads](https://img.shields.io/pepy/dt/ikpls)](https://pepy.tech/project/ikpls/)

[![PyPI - Downloads](https://img.shields.io/pypi/dm/ikpls)](https://pypi.python.org/pypi/ikpls/)

[![License](https://img.shields.io/pypi/l/ikpls)](https://pypi.python.org/pypi/ikpls/)

[![Documentation Status](https://readthedocs.org/projects/ikpls/badge/?version=latest)](https://ikpls.readthedocs.io/en/latest/?badge=latest)

[![Tests Status](https://github.com/Sm00thix/IKPLS/actions/workflows/test_workflow.yml/badge.svg)](https://github.com/Sm00thix/IKPLS/actions/workflows/test_workflow.yml)

[![Test Coverage](https://coveralls.io/repos/github/sm00thix/ikpls/badge.svg?branch=main)](https://coveralls.io/github/sm00thix/ikpls?branch=main)

[![Package Status](https://github.com/Sm00thix/IKPLS/actions/workflows/package_workflow.yml/badge.svg)](https://github.com/Sm00thix/IKPLS/actions/workflows/package_workflow.yml)

[![JOSS Status](https://joss.theoj.org/papers/ac559cbcdc6e6551f58bb3e573a73afc/status.svg)](https://joss.theoj.org/papers/ac559cbcdc6e6551f58bb3e573a73afc)

The `ikpls` software package provides fast and efficient tools for PLS (Partial Least Squares) modeling. This package is designed to help researchers and practitioners handle PLS modeling faster than previously possible - particularly on large datasets.

## Citation
If you use the `ikpls` software package for your work, please cite [this Journal of Open Source Software article](https://joss.theoj.org/papers/10.21105/joss.06533). If you use the fast cross-validation algorithm implemented in `ikpls.fast_cross_validation.numpy_ikpls`, please also cite [this Journal of Chemometrics article](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008).

## Unlock the Power of Fast and Stable Partial Least Squares Modeling with IKPLS

Dive into cutting-edge Python implementations of the IKPLS (Improved Kernel Partial Least Squares) Algorithms #1 and #2 [[1]](#references) for CPUs, GPUs, and TPUs. IKPLS is both fast [[2]](#references) and numerically stable [[3]](#references) making it optimal for PLS modeling.

- Use our NumPy [[4]](#references) based CPU implementations for **seamless integration with
scikit-learn\'s** [[5]](#references) **ecosystem** of machine learning algorithms and pipelines. As the
implementations subclass scikit-learn's BaseEstimator, they can be used with scikit-learn\'s
[cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html).
- Use our JAX [[6]](#references) implementations on CPUs or **leverage powerful GPUs and TPUs for PLS modelling**.
  Our JAX implementations are **end-to-end differentaible** allowing **gradient propagation** when using **PLS as a layer in a deep learning model**.
- Use our combination of IKPLS with Engstrøm's and Jensen's **unbelievably fast cross-validation** algorithm [[7]](#references) to quickly determine the optimal combination of preprocessing and number of PLS components.
- Use any of the above in combination with **sample-weighted PLS** [[8]](#references).
- Use our NumPy or JAX implementations for dimensionality reduction to score space with their respective transform methods.
- Use our NumPy or JAX implementations for reconstruction of original space from score space with their respective inverse_transform methods.

The documentation is available at
<https://ikpls.readthedocs.io/en/latest/>; examples can be found at
<https://github.com/Sm00thix/IKPLS/tree/main/examples>.

## Fast Cross-Validation

In addition to the standalone IKPLS implementations, this package
contains an implementation of IKPLS combined with the novel, fast cross-validation
algorithm by Engstrøm and Jensen [[7]](#references). The fast cross-validation
algorithm benefit both IKPLS Algorithms and especially Algorithm #2. The fast
cross-validation algorithm is mathematically equivalent to the
classical cross-validation algorithm. Still, it is much quicker.
The fast cross-validation algorithm **correctly handles (column-wise)
centering and scaling** of the $\mathbf{X}$ and $\mathbf{Y}$ input matrices using training set means and
standard deviations to avoid data leakage from the validation set. This centering
and scaling can be enabled or disabled independently from eachother and for $\mathbf{X}$ and $\mathbf{Y}$ 
by setting the parameters `center_X`, `center_Y`, `scale_X`, and `scale_Y`, respectively.
In addition to correctly handling (column-wise) centering and scaling,
the fast cross-validation algorithm **correctly handles row-wise preprocessing**
that operates independently on each sample such as (row-wise) centering and scaling
of the $\mathbf{X}$ and $\mathbf{Y}$ input matrices, convolution, or other preprocessing. Row-wise
preprocessing can safely be applied before passing the data to the fast
cross-validation algorithm.

## Prerequisites

The JAX implementations support running on both CPU, GPU, and TPU.

- To enable NVIDIA GPU execution, install JAX and CUDA with:
    ```shell
    pip3 install -U "jax[cuda13]"
    ```

- To enable Google Cloud TPU execution, install JAX with:
    ```shell
    pip3 install -U "jax[tpu]"
    ```

These are typical installation instructions that will be what most users are looking for.
For customized installations, follow the instructions from the [JAX Installation
Guide](https://jax.readthedocs.io/en/latest/installation.html).

To ensure that JAX implementations use float64, set the environment
variable JAX_ENABLE_X64=True as per the [Current
Gotchas](https://github.com/google/jax#current-gotchas).
Alternatively, float64 can be enabled with the following function call:
```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Installation

- Install the package for Python3 using the following command:
    ```shell
    pip3 install ikpls
    ```

- Now you can import the NumPy and JAX implementations with:
    ```python
    from ikpls.numpy_ikpls import PLS as NpPLS
    from ikpls.jax_ikpls_alg_1 import PLS as JAXPLS_Alg_1
    from ikpls.jax_ikpls_alg_2 import PLS as JAXPLS_Alg_2
    from ikpls.fast_cross_validation.numpy_ikpls import PLS as NpPLS_FastCV
    ```

## Quick Start

### Use the ikpls package for PLS modeling

> ```python
> import numpy as np
>
> from ikpls.numpy_ikpls import PLS
>
>
> N = 100  # Number of samples.
> K = 50  # Number of features.
> M = 10  # Number of targets.
> A = 20  # Number of latent variables (PLS components).
>
> X = np.random.uniform(size=(N, K)) # Predictor variables
> Y = np.random.uniform(size=(N, M)) # Target variables
> w = np.random.uniform(size=(N, )) # sample weights
>
> # The other PLS algorithms and implementations have the same interface for fit()
> # and predict(). The fast cross-validation implementation with IKPLS has a
> # different interface.
> np_ikpls_alg_1 = PLS(algorithm=1)
> np_ikpls_alg_1.fit(X, Y, A, w)
>
> # Has shape (A, N, M) = (20, 100, 10). Contains a prediction for all possible
> # numbers of components up to and including A.
> y_pred = np_ikpls_alg_1.predict(X)
>
> # Has shape (N, M) = (100, 10).
> y_pred_20_components = np_ikpls_alg_1.predict(X, n_components=20)
> (y_pred_20_components == y_pred[19]).all()  # True
>
> # The internal model parameters can be accessed as follows:
>
> # Regression coefficients tensor of shape (A, K, M) = (20, 50, 10).
> np_ikpls_alg_1.B
>
> # X weights matrix of shape (K, A) = (50, 20).
> np_ikpls_alg_1.W
>
> # X loadings matrix of shape (K, A) = (50, 20).
> np_ikpls_alg_1.P
>
> # Y loadings matrix of shape (M, A) = (10, 20).
> np_ikpls_alg_1.Q
>
> # X rotations matrix of shape (K, A) = (50, 20).
> np_ikpls_alg_1.R
>
> # Mapping from n_components to Y rotations matrix of shape (M, n_components).
> # This is not required to compute np_ikpls_alg_1.B and is therefore lazily evaluated and cached.
> np_ikpls_alg_1.R_Y
>
> # Y rotations matrix of shape (M, A) = (10, 20)
> np_ikpls_alg_1.R_Y[20] # R_Y is now cached for 20 components.
>
> # Y rotations matrix for 19 components of shape (M, 19) = (10, 19)
> # This is NOT the same as np_ikpls_alg_1.R_Y[20][:, :19]
> np_ikpls_alg_1.R_Y[19] # R_Y is now cached for 20 and 19 components.
>
> # X scores matrix of shape (N, A) = (100, 20).
> # This is only computed for IKPLS Algorithm #1.
> np_ikpls_alg_1.T
> ```

### Examples

In [examples](https://github.com/Sm00thix/IKPLS/tree/main/examples), you
will find:

-   [Fit and Predict with NumPy.](https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_numpy.py)
-   [Fit and Predict with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_jax.py)
-   [Cross-validate with NumPy.](https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_numpy.py)
-   [Cross-validate with NumPy and scikit-learn.](https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_numpy_sklearn.py)
-   [Cross-validate with NumPy and fast cross-validation.](https://github.com/Sm00thix/IKPLS/tree/main/examples/fast_cross_val_numpy.py)
-   [Cross-validate with NumPy and weighted fast cross-validation.](https://github.com/Sm00thix/IKPLS/tree/main/examples/fast_weighted_cross_val_numpy.py)
-   [Cross-validate with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/cross_val_jax.py)
-   [Compute the gradient of a preprocessing convolution filter with respect to the RMSE between the target value and the value predicted by PLS after fitting with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/gradient_jax.py)
-   [Weighted Fit and Predict with NumPy.](https://github.com/Sm00thix/IKPLS/tree/main/examples/weighted_fit_predict_numpy.py)
-   [Weighted Fit and Predict with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/fit_predict_jax.py)
-   [Weighted cross-validation with NumPy.](https://github.com/Sm00thix/IKPLS/tree/main/examples/weighted_cross_val_numpy.py)
-   [Weighted cross-validation with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/weighted_cross_val_jax.py)
-   [Fit, transform to score space, and inverse transform with NumPy.](https://github.com/Sm00thix/IKPLS/tree/main/examples/transform_numpy.py)
-   [Fit, transform to score space, and inverse transform with JAX.](https://github.com/Sm00thix/IKPLS/tree/main/examples/transform_jax.py)

## Changelog
   
See [CHANGELOG.md](CHANGELOG.md) for version history and release notes.

## Contribute

To contribute, please read the [Contribution
Guidelines](https://github.com/Sm00thix/IKPLS/blob/main/CONTRIBUTING.md).

## References

1. [Dayal, B. S. and MacGregor, J. F. (1997). Improved PLS algorithms. *Journal of Chemometrics*, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23?)
2. [Alin, A. (2009). Comparison of PLS algorithms when the number of objects is much larger than the number of variables. *Statistical Papers*, 50, 711-720.](https://doi.org/10.1007/s00362-009-0251-7)
3. [Andersson, M. (2009). A comparison of nine PLS1 algorithms. *Journal of Chemometrics*, 23(10), 518-529.](https://doi.org/10.1002/cem.1248)
4. [NumPy](https://numpy.org/)
5. [scikit-learn](https://scikit-learn.org/stable/)
6. [JAX](https://jax.readthedocs.io/en/latest/)
7. [Engstrøm, O.-C. G. and Jensen, M. H. (2025). Fast Partition-Based Cross-Validation With Centering and Scaling for $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008)
8. [Becker and Ismail (2016). Accounting for sampling weights in PLS path modeling: Simulations and empirical examples. *European Management Journal*, 34(6), 606-617.](https://doi.org/10.1016/j.emj.2016.06.009)


## Funding
- Up until May 31st 2025, this work has been carried out as part of an industrial Ph. D. project receiving funding from [FOSS Analytical A/S](https://www.fossanalytics.com/) and [The Innovation Fund Denmark](https://innovationsfonden.dk/en). Grant number 1044-00108B.
- From June 1st 2025 and onward, this work is sponsored by [FOSS Analytical A/S](https://www.fossanalytics.com/).