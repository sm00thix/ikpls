# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0]
### Added
- A `transform` method for all implementations that transforms $\mathbf{X}$ and/or $\mathbf{Y}$ to score space.
- An `inverse_transform` method for all implementations that reconstructs $\mathbf{X}$ and/or $\mathbf{Y}$ from score space to original space.
- A `fit_transform` method for all implementations that first calls `fit` and then `transform`.
- Tests covering `transform`, `inverse_transform`, and `fit_transform`. // TODO: Add this.
- Examples covering `transform`, `inverse_transform`, and `fit_transform`. // TODO: Add this.
- Downloading the JAX dependencies is now optional and the default download does not include this. // TODO: Add this.

## [3.0.0] - 2025-07-11

### Added
- Fast cross-validation for weighted IKPLS through integration with the `cvmatrix` package.
- Support for all 16 (12 unique) combinations of weighted centering and weighted scaling for $\mathbf{X}$ and $\mathbf{Y}$ matrices.
- Direct dependency on `cvmatrix` software package for enhanced cross-validation capabilities.

### Changed
- Fast cross-validation algorithm now correctly handles weighted cases without increasing time or space complexity.

## [2.0.0] - 2024

### Added
- Sample-weighted PLS functionality using weighted mean and standard deviation as formulated by National Institute of Science and Technology (NIST).
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
- IKPLS Algorithm #1 and #2 implementations based on Dayal and MacGregor (1997) [[1]](#references).
- Numerically stable and fast PLS modeling.
- GPU and TPU acceleration via JAX.
- Compatible with scikit-learn's `cross_validate` function.
- Fast cross-validation based on Engstrøm's and Jensen's algorithm [[2]](#references-1) and implementation [[3]](#references-1).
## References

1. Dayal, B. S. and MacGregor, J. F. (1997). Improved PLS algorithms. *Journal of Chemometrics*, 11(1), 73-85.
2. Engstrøm, O.-C. G. and Jensen, M. H. (2025). Fast Partition-Based Cross-Validation With Centering and Scaling for $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$
3. CVMatrix. Fast computation of possibly weighted and possibly centered/scaled training set kernel matrices in a cross-validation setting.

[Unreleased]: https://github.com/sm00thix/ikpls/compare/v3.0.0.post1...HEAD
[3.0.0]: https://github.com/sm00thix/ikpls/releases/tag/v3.0.0.post1
[2.0.0]: https://github.com/sm00thix/ikpls/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/sm00thix/ikpls/releases/tag/v1.0.0