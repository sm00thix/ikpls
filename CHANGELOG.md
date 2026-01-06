# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-01-05
### Added
- It is now possible to do `import ikpls` followed by `ikpls.submodule` or `import ikpls.submodule` where `submodule` is any of `numpy_ikpls`, `fast_cross_validation`, `jax_ikpls_alg_1`, `jax_ikpls_alg_2`, and `fast_cross_validation.numpy_ikpls`.

### Changed
- The JAX implementations are now optional.
- To install only the NumPy implementations and their dependencies, execute `pip install ikpls`.
- To install both the NumPy and JAX implementations and their dependencies, execute `pip install "ikpls[jax]"`.

## [3.1.0] - 2025-01-05
### Added
- `R_Y` attribute with Y rotations for all PLS implementations except fast cross-validation. `R_Y` is set and reset by fit but not computed until accessed at which point it is cached for future access. `R_Y` is a concrete Mapping accessing the Y rotations for a number of components, nc, is done by `R_Y[nc]` which returns an array of shape `(M, nc)`. Using `R_Y[A][:, :nc]` is generally NOT equivalent to `R_Y[nc]`. Here `M` is the number of columns in `Y`, and `A` is the maximum number of components that the PLS model was fitted with.
- A `transform` method for all implementations that transforms $\mathbf{X}$ and/or $\mathbf{Y}$ to score space.
- An `inverse_transform` method for all implementations that reconstructs $\mathbf{X}$ and/or $\mathbf{Y}$ from score space to original space.
- A `fit_transform` method for all implementations that first calls `fit` and then `transform`.

## [3.0.0] - 2025-07-11

### Added
- Fast cross-validation for weighted IKPLS through integration with the `cvmatrix` package.
- Support for all 16 (12 unique) combinations of weighted centering and weighted scaling for $\mathbf{X}$ and $\mathbf{Y}$ matrices.
- Direct dependency on `cvmatrix` software package for enhanced cross-validation capabilities.

### Changed
- Fast cross-validation algorithm now correctly handles weighted cases without increasing time or space complexity.

## [2.0.0] - 2024

### Added
- Sample-weighted PLS functionality based on Becker and Ismail [[1]](#referemces) using weighted mean and standard deviation as formulated by National Institute of Science and Technology (NIST) [[2,3]](#references).
- Weighted cross-validation methods for both NumPy and JAX implementations via their respective `cross_validate` methods.
- Support for weighted IKPLS modeling across all implementations.

### References
1. [Weighted mean. *National Institute of Standards and Technology*.](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weigmean.pdf)
2. [Weighted standard deviation. *National Institute of Standards and Technology*.](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf)
3. [Becker and Ismail (2016). Accounting for sampling weights in PLS path modeling: Simulations and empirical examples. *European Management Journal*, 34(6), 606-617.](https://doi.org/10.1016/j.emj.2016.06.009)

## [1.0.0] - Initial Release

### Added
- NumPy-based CPU implementations of IKPLS Algorithms #1 and #2.
- JAX implementations supporting CPU, GPU, and TPU execution.
- Integration with scikit-learn's ecosystem (BaseEstimator subclassing).
- End-to-end differentiable JAX implementations for gradient propagation.
- Fast cross-validation algorithm by Engstrøm and Jensen.
- Support for column-wise centering and scaling with proper handling of training/validation splits.
- Support for row-wise preprocessing.
- Comprehensive documentation and examples.

### Features
- IKPLS Algorithm #1 and #2 implementations based on Dayal and MacGregor (1997) [[1]](#references-1).
- Numerically stable and fast PLS modeling.
- GPU and TPU acceleration via JAX.
- Compatible with scikit-learn's `cross_validate` function.
- Fast cross-validation based on Engstrøm's and Jensen's algorithm [[2]](#references-1) and implementation [[3]](#references-1).
## References

1. [Dayal, B. S. and MacGregor, J. F. (1997). Improved PLS algorithms. *Journal of Chemometrics*, 11(1), 73-85.](https://doi.org/10.1002/(SICI)1099-128X(199701)11:1%3C73::AID-CEM435%3E3.0.CO;2-%23?)
2. [Engstrøm, O.-C. G. and Jensen, M. H. (2025). Fast Partition-Based Cross-Validation With Centering and Scaling for $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.70008)
3. [CVMatrix. Fast computation of possibly weighted and possibly centered/scaled training set kernel matrices in a cross-validation setting.](https://github.com/sm00thix/cvmatrix)