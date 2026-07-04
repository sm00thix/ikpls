# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.0.0] - 2026-07-01
### Added
- `ikpls.sklearn.PLS`: a thin, fully scikit-learn-conformant wrapper around the NumPy `ikpls.numpy.PLS`, mirroring `sklearn.cross_decomposition.PLSRegression`. It is both a regressor (`predict` / `score(X, y, sample_weight=None)`) and a dimensionality-reducing transformer (`transform` / `inverse_transform`, each supporting both `X` and `Y`), and passes scikit-learn's `check_estimator` suite with no expected failures. `n_components` is a constructor parameter, so the wrapper drops directly into `cross_validate`, `Pipeline`, and `GridSearchCV`. It exposes the `PLSRegression` fitted attributes `x_weights_`, `y_weights_`, `x_loadings_`, `y_loadings_`, `x_rotations_`, `y_rotations_`, `coef_`, `intercept_`, and `n_features_in_`. Following `PLSRegression` (whose `y_weights_` equals its `y_loadings_`), `y_weights_` equals `y_loadings_`. `coef_` and `intercept_` match `PLSRegression` value-for-value: `intercept_` is the (weighted) `Y`-mean and, as in scikit-learn, `predict` effectively centers `X` (so `predict(X) == (X - X_mean) @ coef_.T + intercept_`). `y_rotations_` is computed lazily on first access (it requires a pseudo-inverse) and cached, so `fit` never pays for it unless it is used. The fast all-components prediction (shape `(A, N, M)`) is preserved via `predict_all_components`, bit-identical to the inner `PLS`. The wrapper performs input validation and plumbing only; all PLS math is delegated unchanged to the inner `PLS`, so the native fast paths (`PLS.fit(X, Y, A)`, `PLS.cross_validate`, and the JAX implementations) are untouched.
- `C` (PLS Y-weights) attribute on the NumPy `ikpls.numpy.PLS` and the JAX `ikpls.jax.PLS` classes (shape `(M, A)`). In Improved Kernel PLS the Y-weights equal the Y-loadings `Q` (this is scikit-learn's `PLSRegression` convention, where `y_weights_` equals `y_loadings_`), so `C` is exposed as a zero-cost alias of `Q`: it adds no computation to `fit` / `stateless_fit` and is never referenced inside a jitted region, so `jax.vmap` and all fast paths are unaffected. Not added to the fast cross-validation classes, which expose no loading/weight matrices.

### Changed
- **Breaking:** Modules were reorganized into a consistent backend-named layout. The NumPy PLS moved from `ikpls.numpy_ikpls` to `ikpls.numpy`; the JAX PLS is now a single `ikpls.jax.PLS(algorithm=1|2)` class (the former `ikpls.jax_ikpls_alg_1`, `ikpls.jax_ikpls_alg_2`, and `ikpls.jax_ikpls_base` modules are now private under `ikpls._impl`); the fast cross-validation modules are now `ikpls.fast_cross_validation.numpy` and `ikpls.fast_cross_validation.jax` (renamed from `numpy_ikpls` / `jax_ikpls`); and the (new) scikit-learn wrapper class is named `PLS` in `ikpls.sklearn`, consistent with the other backends. Update imports to, e.g., `from ikpls.numpy import PLS`, `from ikpls.jax import PLS`, `from ikpls.sklearn import PLS`, `from ikpls.fast_cross_validation.numpy import PLS`, and `from ikpls.fast_cross_validation.jax import PLS`. `ikpls.jax.PLS(algorithm=...)` returns the selected algorithm's implementation while remaining `isinstance` / `issubclass`-compatible with `ikpls.jax.PLS` and exposing `.algorithm`.
- **Breaking:** The default `ddof` for the (weighted) standard deviation used in scaling is now `0` (was `1`) across all PLS classes (`ikpls.numpy`, the JAX classes, and both fast cross-validation classes) and the `ikpls.sklearn.PLS` wrapper. `ddof=0` matches scikit-learn's frequency-weight `sample_weight` semantics, so `ikpls.sklearn.PLS` passes scikit-learn's `sample_weight`-equivalence check with no expected-fail. Note that scikit-learn's own `PLSRegression` scales with `ddof=1`; pass `ddof=1` explicitly to reproduce its scaling. Weighted or scaled fits that relied on the previous `ddof=1` default change numerically; pass `ddof=1` to retain the old behavior.
- **Breaking:** The per-sample weights parameter was renamed from `weights` to `sample_weight` across the entire public API -- `fit`, `fit_transform`, and `cross_validate` on the NumPy (`ikpls.numpy`) and JAX (`ikpls.jax`) PLS classes (plus `stateless_fit` on the JAX classes, which have no NumPy counterpart), and `cross_validate` on both fast cross-validation classes (`ikpls.fast_cross_validation.numpy` / `.jax`) -- to match scikit-learn's `sample_weight` naming convention (already used by the `ikpls.sklearn.PLS` wrapper). Callers that passed `weights=` by keyword must switch to `sample_weight=`; positional calls are unaffected.
- The minimum required `scikit-learn` version is now `>=1.6.0` (was `>=1.5.0`); the conformant wrapper relies on `sklearn.utils.validation.validate_data` and the `__sklearn_tags__` API introduced in 1.6.
- `tqdm` moved from the base dependencies to the `jax` extra, since it is only used by the JAX implementations (the `cross_validate` progress bars). A NumPy-only `pip install ikpls` no longer installs `tqdm`; it is still installed by `pip install ikpls[jax]`.
- The minimum required `cvmatrix` version is now `>=3.2.1` in both the base dependencies (was `>=3.0.0`) and the `jax` extra (was `>=3.2.0`): cvmatrix 3.2.1 no longer imports JAX at import time, which the lazy JAX-optional import layout relies on.

### Removed
- **Breaking:** `ikpls.numpy.PLS` no longer subclasses scikit-learn's `BaseEstimator`. Its native API (`fit(X, Y, A)` with the component count as a fit argument, and `predict` returning all component counts at once) is not scikit-learn-conformant and the `BaseEstimator` inheritance was non-functional for real scikit-learn integration. For use inside the scikit-learn ecosystem, use the new `ikpls.sklearn.PLS` wrapper instead. Code relying on `isinstance(pls, BaseEstimator)`, `sklearn.base.clone(pls)`, or `pls.get_params()` / `set_params()` on the raw NumPy `PLS` must switch to `ikpls.sklearn.PLS`.

### Fixed
- `ikpls.fast_cross_validation.numpy.PLS` (Algorithm #1): calling `cross_validate` WITHOUT `sample_weight` on an instance previously used for a weighted `cross_validate` no longer silently applies the previous call's square-root weights to each fold's training rows; the stale weights are now reset, so a reused instance gives the same results as a fresh one.
- The JAX PLS implementations now degrade gracefully to finite zeros (zero weights/loadings/rotations) instead of producing `NaN` when a component underflows -- e.g. an entirely-constant `Y` with no X-Y covariance (`max_stable_components == 0`). This matches the NumPy implementation and scikit-learn's `PLSRegression` (whose `x_rotations_` is zero in that case). The `jax.lax.scan` fit cannot break out of the component loop as the NumPy loop does, so the weight-norm and `t^T t` divisions are now guarded; stable components are numerically unchanged. The standardization of a constant (zero-variance) feature is likewise guarded before its square root, so reverse-mode gradients through such a fit are finite (forward-mode already was); the standard-deviation values are unchanged for every input real data can produce -- the pre-sqrt clamp differs from the old post-sqrt clamp only for a variance within one ULP of `eps**2`, which is unreachable in practice. (Gradients through a fully-degenerate, entirely-constant-`Y` fit remain undefined in both forward and reverse mode -- the underlying norm/eigendecomposition derivatives at exactly zero / repeated eigenvalues are -- but merely over-extracting components on ordinary data does not trigger this.)

## [5.0.0] - 2026-06-30
### Removed
- **Breaking:** The `differentiable` constructor parameter of the JAX PLS classes (`jax_ikpls_alg_1.PLS`, `jax_ikpls_alg_2.PLS`, and `jax_ikpls_base.PLSBase`) has been removed. The `jax.lax.scan` fit and the fixed-size masked-matmul deflation make the JAX implementations end-to-end reverse-mode differentiable by construction, so the separate non-differentiable code path no longer exists and the flag is obsolete. Code that passed `differentiable=...` must drop the argument.
- **Breaking:** The JAX PLS implementations no longer emit the "weight is close to zero" underflow warning. That warning relied on an ordered host `io_callback` that is incompatible with `jax.vmap` and runs counter to JAX's fast, batched execution model; the JAX fit now runs warning-free and is therefore directly `jax.vmap`-able (e.g. to batch many fits). `max_stable_components` is still tracked: it is now computed on-device (no host callback), so it is reported for a single `fit` and is additionally returned by `stateless_fit` (so it is available per fit under `jax.vmap`). The NumPy implementations are unchanged.

### Added
- JAX fast cross-validation: `ikpls.fast_cross_validation.jax_ikpls.PLS`, a GPU/TPU-friendly counterpart to the NumPy fast cross-validation. It computes per-fold training matrices by rank-update via the `cvmatrix` JAX backend and batches the folds with `jax.vmap`. Supports both Improved Kernel PLS Algorithm #1 and #2 and takes a `batch_size` argument to bound memory. Requires `ikpls[jax]` and `cvmatrix[jax]`.

### Changed
- The JAX fit (`stateless_fit`) now iterates components with `jax.lax.scan` instead of an unrolled Python loop, making the traced graph O(1) in the number of components. Cold (compile-dominated) single fits are dramatically faster at large component counts; warm execution and all numerical results are unchanged.
- The JAX `cross_validate` now batches folds with `jax.vmap` (with a new `batch_size` argument to bound memory) instead of looping over folds in Python, substantially accelerating cross-validation.
- Switched build and development tooling from Poetry to [uv](https://docs.astral.sh/uv/): the project now uses PEP 621 `[project]` metadata, the `hatchling` build backend, and a PEP 735 `[dependency-groups]` dev group (`poetry.lock` is replaced by `uv.lock`). The published package and its runtime dependencies are unchanged, except that the `jax` extra now explicitly requires `cvmatrix>=3.2.0` (the JAX fast cross-validation relies on the cvmatrix JAX backend introduced in that release).

## [4.0.0] - 2026-01-05
### Added
- It is now possible to do `import ikpls` followed by `ikpls.submodule` or `import ikpls.submodule` where `submodule` is any of `numpy_ikpls`, `fast_cross_validation`, `jax_ikpls_alg_1`, `jax_ikpls_alg_2`, and `fast_cross_validation.numpy_ikpls`.

### Changed
- The JAX implementations are now optional.
- To install only the NumPy implementations and their dependencies, execute `pip install ikpls`.
- To install both the NumPy and JAX implementations and their dependencies, execute `pip install "ikpls[jax]"`.

## [3.1.0] - 2026-01-05
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
