import os
from timeit import Timer, default_timer

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from numpy import ndarray
from sklearn.cross_decomposition import PLSRegression as SK_PLS
from sklearn.model_selection import KFold, cross_validate


class SK_PLS_All_Components(SK_PLS):
    """
    Description
    -----------
    Subclass of sklearn's PLSRegression that stores regression matrices for all possible components during fitting.

    Parameters
    ----------
    n_components : int
        Number of components to use in the PLS fit.

    **kwargs:
        Additional keyword arguments to pass to the superclass constructor.
    """

    def __init__(self, n_components, **kwargs):
        super().__init__(n_components=n_components, **kwargs)

    def fit(
        self,
        X,
        Y,
    ):  # Override fit method to store regression matrices for all possible number of components. This is MUCH faster than fitting an entirely new model for every single number of components
        """
        Description
        -----------
        Fit the PLS model and store regression matrices for all possible components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Y : Array of shape (N, M)
            Response variables.
        """
        super().fit(X, Y)
        B = np.empty(
            (self.n_components, self.n_features_in_, self.y_loadings_.shape[0])
        )
        for i in range(B.shape[0]):
            B_at_component_i = np.dot(
                self.x_rotations_[..., : i + 1],
                self.y_loadings_[..., : i + 1].T,
            )
            B[i] = B_at_component_i
        self.B = B

    def predict(self, X: npt.ArrayLike) -> ndarray:
        """
        Description
        -----------
        Predict the output for each number of components up to A.
        This is MUCH faster than calling the predict function
        n_components number of times with different number of components.

        Parameters
        ----------
        X : Array of shape (N, K)
            Predictor variables.

        Returns:
        Y_pred : Array of shape (A, N, M)
            Predicted response variables for each number of components up to A.
        """
        return (X - self._x_mean) / self._x_std @ self.B + self.intercept_


def gen_random_data(N, K, M):
    """
    Description
    -----------
    Generate random data for testing.

    Parameters
    ----------
    N : int
        Number of samples.

    K : int
        Number of features in X.
    M : int
        Number of features in Y.

    Returns
    -------
    (X, Y) : Tuple of arrays
        Randomly generated input X and target Y.
    """
    rng = np.random.default_rng(seed=42)
    X = rng.random((N, K), dtype=np.float64)
    Y = rng.random((N, M), dtype=np.float64)
    return X, Y


def mse_for_each_target(estimator, X, Y_true, **kwargs):
    """
    Description
    -----------
    Calculate lowest mean squared error (mse) for each target and the corresponding number of components that minimizes the mse.

    Parameters
    ----------
    estimator : Estimator object
        PLS estimator with a 'predict' method.

    X : Array of shape (N, K)
        Predictor variables.

    Y_true : Array of shape (N, M)
        True response variables.

    **kwargs:
        Additional keyword arguments to pass to the 'predict' method.

    Returns
    -------
    metrics: dict
        Dictionary containing MSE and number of components for each target.
    """

    # Y_true has shape (N, M).
    Y_pred = estimator.predict(X, **kwargs)  # Shape (A, N, M).
    e = Y_true - Y_pred  # Shape (A, N, M).
    se = e**2  # Shape (A, N, M).
    mse = np.mean(se, axis=-2)  # Compute the mean over samples. Shape (A, M).
    row_idxs = np.argmin(
        mse, axis=0
    )  # The number of components that minimizes the MSE for each target. Shape (M,).
    lowest_mses = mse[
        row_idxs, np.arange(mse.shape[1])
    ]  # The lowest MSE for each target. Shape (M,).
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    mse_names = [
        f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]  # List of names for the lowest MSE values.
    num_components_names = [  # List of names for the number of components that achieves the lowest MSE for each target.
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names  # List of all names.
    all_values = np.concatenate((lowest_mses, num_components))  # Array of all values.
    return dict(zip(all_names, all_values))


def mse_for_each_target_fast_cv(Y_true, Y_pred):
    """
    Description
    -----------
    Calculate lowest mean squared error (mse) for each target and the corresponding number of components that minimizes the mse.

    Parameters
    ----------
    Y_true : Array of shape (N, M)
        True response variables.

    Y_pred : Array of shape (A, N, M)
        Predicted response variables.

    Returns
    -------
    metrics: dict
        Dictionary containing MSE and number of components for each target.
    """

    # Y_true has shape (N, M).
    e = Y_true - Y_pred  # Shape (A, N, M).
    se = e**2  # Shape (A, N, M).
    mse = np.mean(se, axis=-2)  # Compute the mean over samples. Shape (A, M).
    row_idxs = np.argmin(
        mse, axis=0
    )  # The number of components that minimizes the MSE for each target. Shape (M,).
    lowest_mses = mse[
        row_idxs, np.arange(mse.shape[1])
    ]  # The lowest MSE for each target. Shape (M,).
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    mse_names = [
        f"lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]  # List of names for the lowest MSE values.
    num_components_names = [  # List of names for the number of components that achieves the lowest MSE for each target.
        f"num_components_lowest_mse_target_{i}" for i in range(lowest_mses.shape[0])
    ]
    all_names = mse_names + num_components_names  # List of all names.
    all_values = np.concatenate((lowest_mses, num_components))  # Array of all values.
    return dict(zip(all_names, all_values))


def jax_mse_for_each_target(Y_true, Y_pred):
    """
    Calculate mean squared error for each target and the corresponding number of components using JAX.

    Parameters
    ----------
    Y_true : Array of shape (N, M)
        True target values.

    Y_pred : Array of shape (A, N, M)
        Predicted target values.

    Returns
    -------
    all_values : Array of shape (2*M,)
    """
    # Y_true has shape (N, M)
    e = Y_true - Y_pred  # Shape (A, N, M)
    se = e**2  # Shape (A, N, M)
    mse = jnp.mean(se, axis=-2)  # Compute the mean over samples. Shape (A, M).
    row_idxs = jnp.argmin(  # The number of components that minimizes the MSE for each target. Shape (M,).
        mse, axis=0
    )
    lowest_mses = mse[  # The lowest MSE for each target. Shape (M,).
        row_idxs, jnp.arange(mse.shape[1])
    ]  # The lowest MSE for each target.
    num_components = (
        row_idxs + 1
    )  # Indices are 0-indexed but number of components is 1-indexed.
    all_values = jnp.concatenate((lowest_mses, num_components))  # Array of all values.
    return all_values


def jax_metric_names(M):
    """
    Description
    -----------
    Generate metric names for MSE and number of components.

    Parameters
    ----------
    M : int
        Number of targets.

    Returns
    -------
    all_names : list[str]
        List of metric names.
    """
    mse_names = [
        f"lowest_mse_target_{i}" for i in range(M)
    ]  # List of names for the lowest MSE values.
    num_components_names = [
        f"num_components_lowest_mse_target_{i}" for i in range(M)
    ]  # List of names for the number of components that achieves the lowest MSE for each target.
    all_names = mse_names + num_components_names  # List of all names.
    return all_names


def cross_val_cpu_pls(pls, X, Y, n_splits, params, n_jobs, verbose, estimate):
    """
    Description
    -----------
    Perform cross-validation for PLS on CPU and measure the execution time.

    Parameters
    ----------
    pls : Estimator object
        PLS estimator with a 'predict' method.
    X : Array of shape (N, K)
        Predictor variables.
    Y : Array of shape (N, M)
        Response variables.
    n_splits : int
        Number of splits in cross-validation.
    params : dict
        Parameters to pass to the fit method.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int
        Verbosity level.
    estimate : bool
        Whether to estimate the time taken to do a full cross-validation based on the
        time taken to execute only 2*n_jobs cross-validation splits.

    Returns
    -------
    time : float
        Execution time for cross-validation.
    """
    cv = KFold(n_splits=n_splits, shuffle=False)
    cv_to_pass = cv
    if estimate:
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        n_splits_to_execute = 2 * n_jobs
        if n_splits_to_execute > n_splits:
            n_splits_to_execute = n_splits

        def cv_estimate():
            for _ in range(n_splits_to_execute):
                (train, test) = next(cv.split(X))
                yield train, test

        cv_to_pass = cv_estimate()
    t = Timer(
        stmt="scores = cross_validate(pls, X, Y, cv=cv_to_pass, scoring=mse_for_each_target, return_estimator=False, params=params, n_jobs=n_jobs, verbose=verbose, )",
        timer=default_timer,
        globals=locals() | globals(),
    )
    time_taken = t.timeit(number=1)
    if estimate:
        time_taken *= n_splits / n_splits_to_execute
        print(f"Estimated time for full cross-validation: {time_taken:.2f} seconds.")
    return time_taken


def fast_cross_val_cpu_pls(pls, X, Y, A, n_splits, n_jobs, verbose):
    """
    Description
    -----------
    Perform cross-validation for Fast IKPLS CV on CPU and measure the execution time.

    Parameters
    ----------
    pls : Estimator object
        PLS estimator with a 'predict' method.
    X : Array of shape (N, K)
        Predictor variables.
    Y : Array of shape (N, M)
        Response variables.
    n_splits : int
        Number of splits in cross-validation.
    params : dict
        Parameters to pass to the fit method.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int
        Verbosity level.

    Returns
    -------
    time : float
        Execution time for cross-validation.
    """
    cv_splits = np.zeros(X.shape[0])
    for i in range(X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits + 1
        cv_splits[i * split_size : (i + 1) * split_size] = i
    prev_max_idx = (X.shape[0] % n_splits) * (X.shape[0] // n_splits + 1)
    for i in range(n_splits - X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits
        cv_splits[
            prev_max_idx + i * split_size : prev_max_idx + (i + 1) * split_size
        ] = (i + X.shape[0] % n_splits)
    t = Timer(
        stmt="scores = pls.cross_validate(X=X, Y=Y, A=A, cv_splits=cv_splits, metric_function=mse_for_each_target_fast_cv, n_jobs=n_jobs, verbose=verbose, )",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def single_fit_cpu_pls(pls, X, Y, params=None):
    """
    Description
    -----------
    Fit PLS model on CPU and measure the execution time.

    Parameters
    ----------
    pls : Estimator object
        PLS estimator with a 'predict' method.
    X : Array of shape (N, K)
        Predictor variables.
    Y : Array of shape (N, M)
        Response variables.
    params : dict
        Parameters to pass to the fit method.

    Returns
    -------
    time : float
        Execution time for single fit.
    """
    t = Timer(
        stmt="pls.fit(X, Y, **params)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def jax_preprocessing_function(X_train, Y_train, X_val, Y_val):
    """
    Description
    -----------
    Preprocessing function for JAX PLS.

    Parameters
    ----------
    X_train : Array of shape (N_train, K)
        Training predictor variables.
    Y_train : Array of shape (N_train, M)
        Training response variables.
    X_val : Array of shape (N_val, K)
        Validation predictor variables.
    Y_val : Array of shape (N_val, M)
        Validation response variables.

    Returns
    -------
    X_train : Array of shape (N_train, K)
        Preprocessed training predictor variables.
    Y_train : Array of shape (N_train, M)
        Preprocessed training response variables.
    X_val : Array of shape (N_val, K)
        Preprocessed validation predictor variables.
    Y_val : Array of shape (N_val, M)
        Preprocessed validation response variables.
    """
    return X_train, Y_train, X_val, Y_val


def cross_val_gpu_pls(pls, X, Y, n_components, n_splits, show_progress):
    """
    Description
    -----------
    Perform cross-validation for PLS on GPU and measure the execution time.

    Parameters
    ----------
    pls : Estimator object
        PLS estimator with a 'predict' method.
    X : Array of shape (N, K)
        Predictor variables.
    Y : Array of shape (N, M)
        Response variables.
    n_components : int
        Number of components.
    n_splits : int
        Number of splits in cross-validation.
    show_progress : bool
        Whether to show progress.

    Returns
    -------
    time : float
        Execution time for cross-validation.
    """
    cv_splits = np.zeros(X.shape[0])
    for i in range(X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits + 1
        cv_splits[i * split_size : (i + 1) * split_size] = i
    prev_max_idx = (X.shape[0] % n_splits) * (X.shape[0] // n_splits + 1)
    for i in range(n_splits - X.shape[0] % n_splits):
        split_size = X.shape[0] // n_splits
        cv_splits[
            prev_max_idx + i * split_size : prev_max_idx + (i + 1) * split_size
        ] = (i + X.shape[0] % n_splits)
    t = Timer(
        stmt="pls.cross_validate(X=X, Y=Y, A=n_components, cv_splits=cv_splits, preprocessing_function=jax_preprocessing_function, metric_function=jax_mse_for_each_target, metric_names=jax_metric_names(Y.shape[1]), show_progress=show_progress)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)


def single_fit_gpu_pls(pls, X, Y, n_components):
    """
    Description
    -----------
    Fit PLS model on GPU and measure the execution time.

    Parameters
    ----------
    pls : Estimator object
        PLS estimator with a 'predict' method.
    X : Array of shape (N, K)
        Predictor variables.
    Y : Array of shape (N, M)
        Response variables.
    n_components : int
        Number of components.

    Returns
    -------
    time : float
        Execution time for single fit.
    """
    t = Timer(
        stmt="pls.fit(X, Y, A=n_components)",
        timer=default_timer,
        globals=locals() | globals(),
    )
    return t.timeit(number=1)
