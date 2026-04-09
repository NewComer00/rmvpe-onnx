import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from rmvpe_onnx import __version__

project = 'rmvpe-onnx'
copyright = '2026, NewComer00'
author = 'NewComer00'
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxarg.ext",  # placed before "sphinx_autodoc_typehints", see sphinx issue #14333
    "sphinx_autodoc_typehints",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['_static']
