# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys

sys.path.insert(0, os.path.abspath("../../adda"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "adda"
copyright = "2026, m-dml team"
author = "m-dml team"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "seed_intersphinx_mapping",
    "sphinx_typo3_theme",
]

add_module_names = True
toc_object_entries_show_parents = "hide"
templates_path = ["_templates"]
exclude_patterns = ["../../adda/_version.py"]
autosummary_generate = True


# --- Autodoc ---------------------------------------------------------------


autodoc_default_options = {
    "members": True,
    "member-order": "groupwise",
    "undoc-members": True,
    "exclude-members": "__weakref__, __init__",
    "show-inheritance": True,
}

autoclass_content = "class"
ignore_module_all = False
autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_typo3_theme"

# -- Intersphinx options
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
}

html_logo = "images/m-dml_logo_banner_cropped.png"
html_favicon = "images/m-dml_logo_dark.png"

# -- Custom scripts ----------------------------------------------------------


def cleanup():
    """Delete the _autosummary folder before building the documentation."""
    if not os.path.exists(os.path.join(os.getcwd(), "source", "_autosummary")):
        return
    print(f"Cleaning up _autosummary folder in {os.getcwd()}")
    shutil.rmtree(os.path.join(os.getcwd(), "source", "_autosummary"))


def change_content_of_main_generated_index(app, what, name, obj, options, lines):
    """Add text to the main index page."""
    if what == "module":
        if name == "da_tools":
            lines.insert(0, "Api reference for da_tools-tools.")
            print("")


def setup(app):
    cleanup()
    app.connect("autodoc-process-docstring", change_content_of_main_generated_index)
