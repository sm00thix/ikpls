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
}


def maybe_skip_member(app, what, name, obj, skip, options):
    if name in [
        "set_fit_request",
        "set_predict_request",
        "set_transform_request",
        "set_inverse_transform_request",
        "_abc_impl",
    ]:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", maybe_skip_member)
