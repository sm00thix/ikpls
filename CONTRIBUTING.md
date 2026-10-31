# Contribution Guidelines

We welcome any kind of contribution to our software, from a simple comment or question to a full-fledged [pull request](https://help.github.com/articles/about-pull-requests/).

A contribution can be one of the following cases:

1. You have a question;
2. You think you may have found a bug (including unexpected behavior);
3. You want to change the code base (e.g., to fix a bug, add a new feature, or update documentation).

The sections below outline the steps in each case.

## You have a question

1. Use the search functionality [here](https://github.com/Sm00thix/IKPLS/issues) to see if someone already filed the same issue;
2. If your issue search did not yield any relevant results, make a new issue;
3. Apply the `question` label; apply other labels when relevant.

## You think you may have found a bug

1. Use the search functionality [here](https://github.com/Sm00thix/IKPLS/issues) to see if someone already filed the same issue;
2. If your issue search does not yield any relevant results, make a new issue, providing enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
3. Apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. (**Important**) Announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
2. (**Important**) Wait until some kind of consensus is reached about your idea being a good idea;
3. If needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
4. Make sure the existing tests still work by following the instructions in [Run the test suite](#run-the-test-suite);
5. Add your own tests (if necessary);
6. If you added your own tests, make sure they pass by following the instructions in [Run the test suite](#run-the-test-suite);
7. Lint your changes by following the instructions in [Lint](#lint);
8. If your contribution is a performance enhancement, make sure to include benchmarks. See [Benchmarking](#benchmarking) for more information;
9. Update or expand the documentation;
10. Make sure the documentation builds without errors or warnings by following the instructions in [Build the documentation](#build-the-documentation);
11. Describe your change in [CHANGELOG.md](https://github.com/Sm00thix/IKPLS/blob/main/CHANGELOG.md), under the section for the upcoming version;
12. Check that you can build the package locally and that it passes twine check. See [Build from source](#build-from-source) for more information;
13. [Push](http://rogerdudler.github.io/git-guide/) your feature branch to (your fork of) the IKPLS repository on GitHub;
14. Create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

If you feel like you've made a valuable contribution, but you don't know how to write or run tests for it or generate the documentation, don't let this discourage you from making the pull request; we can help you! Just submit the pull request, but remember that you might be asked to append additional commits to your pull request.

## Build from source

IKPLS uses [uv](https://docs.astral.sh/uv/) to manage its dependencies and packaging. To build the package from source, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/Sm00thix/IKPLS.git
    ```

2. Change to the repository directory:

    ```shell
    cd IKPLS
    ```

3. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

4. Build the package. uv builds in an isolated environment, so no separate dependency-install step is needed:

    ```shell
    uv build
    ```

5. Check the package with twine:

    ```shell
    uvx twine check dist/*
    ```

The source distribution contains every file in your working tree that is not excluded by the
root `.gitignore`, including untracked ones. Build from a clean checkout if you intend to
distribute the result.

If you change the project's dependencies in `pyproject.toml`, refresh the committed lockfile and commit the updated `uv.lock`:

```shell
uv lock
```

## Run the test suite

IKPLS supports Python 3.10 to 3.15, and CI runs the test suite on all of them, on Linux,
Windows and macOS, both with and without JAX. CI also runs it on free-threaded Python 3.15t
(without JAX on Windows, where jaxlib has no free-threaded build) and fails if importing a
test dependency re-enables the GIL. The JAX test suite (`tests/test_ikpls.py`) downloads a
dataset from [IKPLSTestData](https://github.com/Sm00thix/IKPLSTestData), so it needs an
internet connection, and takes about an hour.

To run the test suite, follow these steps:

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

    ```shell
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Install the project together with its development dependencies. To run the full suite (including the JAX implementations), include the `jax` extra:

    ```shell
    uv sync --extra jax --group dev
    ```

    To test the numpy-only installation instead, omit the extra:

    ```shell
    uv sync --group dev
    ```

    To use a specific Python version, add e.g. `--python 3.15`, or `--python 3.15t` for the free-threaded build (uv downloads it if needed).

3. Now, the tests can be run. With the JAX extra installed, run the full suite except the numpy-only tests (which intentionally assert JAX is absent):

    ```shell
    uv run pytest tests --ignore=tests/test_numpy_only.py --typeguard-packages=ikpls --cov=ikpls --cov-report=xml --cov-report=html
    ```

    For a numpy-only installation, run only the numpy-only tests:

    ```shell
    uv run pytest tests/test_numpy_only.py --cov=ikpls --cov-report=xml --cov-report=html
    ```

    By default, the JAX tests run with JAX's JIT compiler disabled, as in CI, where the
    compiled programs would exhaust the runners' memory. To also exercise the compiled code
    paths, set `IKPLS_TEST_JIT=1`:

    ```shell
    IKPLS_TEST_JIT=1 uv run pytest tests --ignore=tests/test_numpy_only.py
    ```

## Lint

The package is linted with [ruff](https://docs.astral.sh/ruff/) in CI. Reproduce the lint gate locally before pushing (this check must pass — it catches syntax errors, undefined names, etc.):

```shell
uvx ruff check ikpls/ --select=E9,F63,F7,F82 --output-format=full
```

CI additionally reports style statistics (line length, complexity), which do not fail the build:

```shell
uvx ruff check ikpls/ --select=E501,C901 --line-length=88 --exit-zero --statistics
```

## Build the documentation

The API reference is generated by importing IKPLS, so the documentation must be built in an
environment with IKPLS and its dependencies (including JAX) installed:

1. Install the project with the JAX extra (see [Run the test suite](#run-the-test-suite)):

    ```shell
    uv sync --extra jax --group dev
    ```

2. Build the documentation with sphinx, the sphinx-rtd-theme, and the MyST-Parser:

    ```shell
    uv run --with sphinx --with sphinx-rtd-theme --with myst-parser make -C docs html
    ```

The HTML is written to `docs/build/html`. Check the output for warnings: if an IKPLS module or one of its
dependencies (e.g. JAX) cannot be imported, sphinx still exits successfully, but with
`autodoc: failed to import` warnings and empty API pages.

## Benchmarking

To run benchmarks, follow the instructions [here](https://github.com/Sm00thix/IKPLS/blob/main/paper/README.md).

## Releasing (maintainers)

1. Bump the version in both `pyproject.toml` and `ikpls/__init__.py`, and run `uv lock`
   (`uv.lock` records the version too). The versions must match: the package metadata and
   GitHub release use `pyproject.toml`, the documentation `ikpls/__init__.py`.
2. Give the version's section in [CHANGELOG.md](https://github.com/Sm00thix/IKPLS/blob/main/CHANGELOG.md) its release date.
3. Merge to `main`. When the tests pass on `main`, CI builds the tested commit, publishes
   it to PyPI, and creates a GitHub release `v<version>`, tagged on that commit, with the
   Sigstore-signed distributions attached. Re-publishing a version that is already on PyPI skips the upload, but the
   GitHub-release job then fails because the release already exists.
