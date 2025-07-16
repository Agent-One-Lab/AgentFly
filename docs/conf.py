# docs/conf.py
import os, sys, pathlib

agent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agents")
sys.path.insert(0, agent_path)

project   = "AgentFly"
author    = "AgentFly Team"
copyright = "2025, AgentFly Team"

extensions = [
    "myst_parser",            # parse *.md as MyST Markdown
    "sphinx.ext.autodoc",     # import & document your Python API
    "sphinx.ext.autosummary", # tables of functions/classes
    "sphinx.ext.napoleon",    # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",    # add “[source]” links
    "sphinx.ext.autosectionlabel",
]

# recognise both .md and .rst
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme       = "sphinx_rtd_theme"
html_static_path = ["_static"]
