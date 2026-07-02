# conf.py

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "IKPLS"
copyright = "2023, Ole-Christian Galbo Engstrøm, Erik Schou Dreier, Birthe Møller Jespersen, and Kim Steenstrup Pedersen"
author = "Ole-Christian Galbo Engstrøm"

import ikpls

release = ikpls.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_parser",
]

myst_enable_extensions = [
    "dollarmath",  # Enable dollar sign as a delimiter for math
    # Add other MyST extensions you might need
]

myst_heading_anchors = 2

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = [".rst", ".md"]
master_doc = "index"

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

# -- Extension configuration -------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# -- Options for autodoc extension -------------------------------------------

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "private-members": False,
    # Hide noisy auto-generated members from the documented classes:
    # scikit-learn's metadata-routing setters generated on the
    # BaseEstimator-derived ``ikpls.sklearn.PLS`` wrapper (``fit``/``score`` expose
    # extra metadata, so ``set_fit_request``/``set_score_request`` appear; the
    # other ``set_*_request`` names are listed defensively and are harmless
    # no-ops when not generated), and the ``_abc_impl`` bookkeeping attribute on
    # the JAX ``abc.ABC`` base. (Replaces an older ``autodoc-skip-member`` hook;
    # the inner NumPy ``PLS`` no longer subclasses ``BaseEstimator``.)
    #
    # ``y_rotations_`` (sklearn wrapper) is the one fitted value exposed as a
    # ``@property`` member; every other fitted attribute (including ``C``) is a plain
    # instance attribute documented via a numpydoc ``Attributes`` section (the class
    # docstring for the wrapper, the ``fit`` docstring for the NumPy/JAX classes).
    # Excluding the property member keeps it documented exactly once, consistently
    # with its siblings, and avoids a duplicate-object warning.
    "exclude-members": (
        "set_fit_request,set_predict_request,set_score_request,"
        "set_transform_request,set_inverse_transform_request,_abc_impl,"
        "y_rotations_"
    ),
}
