# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
#
# sys.path.insert(0, os.path.abspath('.'))

from meddocan import __version__

# -- Project information -----------------------------------------------------

project = 'Meddocan'
copyright = '2022, Guillaume Gelabert'
author = 'Guillaume Gelabert'

# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx.ext.autodoc',
    'recommonmark',  # For conversion from markdown to html.
    'sphinx.ext.napoleon',  # For google style docs.
    'sphinx.ext.autosummary',  # To add first docstring line.
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',  # Add links to highlighted source code.
]

# build the templated autosummary files
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
# source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# RestructuredText that will be include at the beginning of every source file.
rst_prolog = """
.. |Doc| replace:: `Doc <https://spacy.io/api/doc>`__
.. |Token| replace:: `Token <https://spacy.io/api/token>`__
.. |Span| replace:: `Span <https://spacy.io/api/span>`__
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Theme options
#
html_theme_options = {
    'github_user': 'GuiGel',
    'github_repo': 'MedDocAn',
    'description': 'Model trained on the Meddocan corpus',
    'sidebar_collapse': True
}

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.8/', None),
    'numpy': ('https://numpy.org/doc/stable', None),
}
