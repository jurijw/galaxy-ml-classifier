# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # add project root

# -- Project information -----------------------------------------------------

project = 'galaxy-ml-classifier'
author = 'Jurij W.'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',    # Extract docstrings
    'sphinx.ext.napoleon',   # Support Google style docstrings
    'sphinx.ext.viewcode',   # Add links to source code
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
