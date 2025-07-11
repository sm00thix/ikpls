import argparse
import os

import jax
from timings.timings import (
    SK_PLS_All_Components,
    cross_val_cpu_pls,
    cross_val_gpu_pls,
    fast_cross_val_cpu_pls,
    gen_random_data,
    single_fit_cpu_pls,
    single_fit_gpu_pls,
)

from ikpls.fast_cross_validation.numpy_ikpls import PLS as NP_PLS_FCV
from ikpls.jax_ikpls_alg_1 import PLS as JAX_PLS_Alg_1
from ikpls.jax_ikpls_alg_2 import PLS as JAX_PLS_Alg_2
from ikpls.numpy_ikpls import PLS as NP_PLS

jax.config.update("jax_enable_x64", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        type=str,
        help="Model to use. Must be either 'sk', 'np1', 'np2', 'fastnp1, 'fastnp2', 'jax1', 'jax2', 'diffjax1', 'diffjax2'. Note that fastnp1 and fastnp2 only support cross-validation and not single fits.",
    )
    parser.add_argument("-n_components", type=int, help="Number of components to use.")
    parser.add_argument(
        "-n_splits",
        type=int,
        help="Number of splits to use in cross-validation. If 1, will perform a single fit.",
    )
    parser.add_argument(
        "-n_jobs",
        type=int,
        help="Number of parallel jobs to use. A value of -1 will use all available cores. Not used for JAX implementations as it is assumed these will run on a TPU or GPU.",
    )
    parser.add_argument("-n", type=int, help="Number of samples to generate.")
    parser.add_argument("-k", type=int, help="Number of features to generate.")
    parser.add_argument("-m", type=int, help="Number of targets to generate.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="timings/user_timings.csv",
        help="Output file to write timings to.",
    )
    parser.add_argument(
        "--estimate",
        dest="estimate",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Estimate the time taken to do a full cross-validation based on the time taken to execute only 2*n_jobs cross-validation splits.",
    )
    args = parser.parse_args()
    config = vars(args)
    model = config["model"]
    n_components = config["n_components"]
    n_splits = config["n_splits"]
    n = config["n"]
    k = config["k"]
    m = config["m"]
    estimate = config["estimate"]

    X, Y = gen_random_data(n, k, m)
    if "jax" in model:
        if model == "jax1":
            pls = JAX_PLS_Alg_1(
                center_X=False, center_Y=False, scale_X=False, scale_Y=False
            )
            name = "JAX Improved Kernel PLS Algorithm #1"
        elif model == "jax2":
            pls = JAX_PLS_Alg_2(
                center_X=False, center_Y=False, scale_X=False, scale_Y=False
            )
            name = "JAX Improved Kernel PLS Algorithm #2"
        elif model == "diffjax1":
            pls = JAX_PLS_Alg_1(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                differentiable=True,
            )
            name = (
                "JAX Improved Kernel PLS Algorithm #1 (backwards mode differentiable)"
            )
        elif model == "diffjax2":
            pls = JAX_PLS_Alg_2(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                reverse_differentiable=True,
            )
            name = (
                "JAX Improved Kernel PLS Algorithm #2 (backwards mode differentiable)"
            )
        if n_splits == 1:
            print(
                f"Fitting {name} with {n_components} components on {n} samples with {k} features and {m} targets."
            )
            time = single_fit_gpu_pls(pls, X, Y, n_components)
        else:
            if estimate:
                raise ValueError("Estimation is not supported for JAX implementations.")
            print(
                f"Fitting {name} with {n_components} components using {n_splits}-fold cross-validation on {n} samples with {k} features and {m} targets."
            )
            time = cross_val_gpu_pls(
                pls, X, Y, n_components, n_splits, show_progress=True
            )
        print(f"Time: {time}")
    else:
        n_jobs = config["n_jobs"]
        if model == "sk":
            pls = SK_PLS_All_Components(n_components=n_components)
            params = {} if n_splits == 1 else None
            name = "scikit-learn NIPALS"
        elif model == "np1":
            pls = NP_PLS(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                algorithm=1,
            )
            params = {"A": n_components}
            name = "NumPy Improved Kernel PLS Algorithm #1"
        elif model == "np2":
            pls = NP_PLS(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                algorithm=2,
            )
            params = {"A": n_components}
            name = "NumPy Improved Kernel PLS Algorithm #2"
        elif model == "fastnp1":
            pls = NP_PLS_FCV(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                algorithm=1,
            )
            name = "NumPy Improved Kernel PLS Algorithm #1 (fast cross-validation)"
        elif model == "fastnp2":
            pls = NP_PLS_FCV(
                center_X=False,
                center_Y=False,
                scale_X=False,
                scale_Y=False,
                algorithm=2,
            )
            name = "NumPy Improved Kernel PLS Algorithm #2 (fast cross-validation)"
        else:
            raise ValueError(
                f"Unknown model: {model}. Must be one of 'sk', 'np1', 'np2', 'fastnp1', 'fastnp2', 'jax1', 'jax2', 'diffjax1', 'diffjax2'."
            )

        if n_splits == 1:
            print(
                f"Fitting {name} with {n_components} components on {n} samples with {k} features and {m} targets."
            )
            time = single_fit_cpu_pls(pls, X, Y, params)
        else:
            fitting_or_estimating = "Estimating" if estimate else "Fitting"
            print(
                f"{fitting_or_estimating} {name} with {n_components} components using {n_splits}-fold cross-validation on {n} samples with {k} features and {m} targets. Using {n_jobs} concurrent workers."
            )
            if model.startswith("fast"):
                if estimate:
                    raise ValueError(
                        "Estimation is not supported for fast cross-validation."
                    )
                time = fast_cross_val_cpu_pls(
                    pls,
                    X,
                    Y,
                    n_components,
                    n_splits=n_splits,
                    n_jobs=n_jobs,
                    verbose=100,
                )
            else:
                time = cross_val_cpu_pls(
                    pls,
                    X,
                    Y,
                    n_splits,
                    params,
                    n_jobs=n_jobs,
                    verbose=1,
                    estimate=estimate,
                )
        print(f"Time: {time}")

    if "jax" in model:
        num_cores = "GPU"
    else:
        num_cores = os.cpu_count() if n_jobs == -1 else n_jobs

    # don't repeat yourself
    if not os.path.exists(args.output):
        with open(args.output, "w") as f:
            f.write("model,n_components,n_splits,n,k,m,time,inferred,njobs\n")
    with open(args.output, "a") as f:
        f.write(
            f"{model},{n_components},{n_splits},{n},{k},{m},{time},{estimate},{num_cores}\n"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        exit(1)
