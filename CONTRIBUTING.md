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
3. Apply the "Question" label; apply other labels when relevant.

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
10. Make sure the documentation builds without errors by following the instructions in [Build the documentation](#build-the-documentation);
11. Check that you can build the package locally and that it passes twine check. See [Build from source](#build-from-source) for more information;
12. [Push](http://rogerdudler.github.io/git-guide/) your feature branch to (your fork of) the IKPLS repository on GitHub;
13. Create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

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

If you change the project's dependencies in `pyproject.toml`, refresh the committed lockfile and commit the updated `uv.lock`:

```shell
uv lock
```

## Run the test suite

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

3. Now, the tests can be run. With the JAX extra installed, run the full suite except the numpy-only tests (which intentionally assert JAX is absent):

    ```shell
    uv run pytest tests --ignore=tests/test_numpy_only.py --typeguard-packages=ikpls --cov=ikpls --cov-report=xml --cov-report=html
    ```

    For a numpy-only installation, run only the numpy-only tests:

    ```shell
    uv run pytest tests/test_numpy_only.py --cov=ikpls --cov-report=xml --cov-report=html
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

1. Install sphinx, the sphinx-rtd-theme, and the MyST-Parser:

    ```shell
    pip3 install sphinx sphinx-rtd-theme myst-parser
    ```

2. Change to the docs directory:

    ```shell
    cd docs
    ```

3. Build the documentation:

    ```shell
    make html
    ```

## Benchmarking

To run benchmarks, follow the instructions [here](https://github.com/Sm00thix/IKPLS/blob/main/paper/README.md).
