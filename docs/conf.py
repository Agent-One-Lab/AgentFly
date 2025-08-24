# docs/conf.py
import os, sys, pathlib
import inspect
import importlib

# Add both project root and agents directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

project   = "AgentFly"
author    = "AgentFly Team"
copyright = "2025, AgentFly Team"

extensions = [
    "myst_parser",            # parse *.md as MyST Markdown
    "sphinx.ext.autodoc",     # import & document your Python API
    "sphinx.ext.autosummary", # tables of functions/classes
    "sphinx.ext.napoleon",    # Google/NumPy-style docstrings
    "sphinx.ext.viewcode",    # add "[source]" links
    "sphinx.ext.autosectionlabel",
]

# recognise both .md and .rst
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme       = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Configure autodoc to include special methods
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__, __call__',
    'exclude-members': '__weakref__'
}

# Custom extension to handle tool documentation
def setup(app):
    from docutils import nodes
    from docutils.parsers.rst import Directive
    
    class ToolDocstringDirective(Directive):
        has_content = True
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = {}
        
        def run(self):
            tool_name = self.arguments[0]
            
            # Import the original function modules
            try:
                if tool_name == "code_interpreter":
                    from agentfly.tools.src.code.tools import code_interpreter
                    original_func = code_interpreter.user_func
                elif tool_name == "google_search_serper":
                    from agentfly.tools.src.search.google_search import google_search_serper
                    original_func = google_search_serper.user_func
                elif tool_name == "answer":
                    from agentfly.tools.src.react.tools import answer
                    original_func = answer.user_func
                else:
                    return [nodes.paragraph(text=f"Tool {tool_name} not found")]
                
                # Get the original docstring
                docstring = inspect.getdoc(original_func)
                if docstring:
                    return [nodes.literal_block(docstring, docstring)]
                else:
                    return [nodes.paragraph(text=f"No docstring found for {tool_name}")]
                    
            except Exception as e:
                return [nodes.paragraph(text=f"Error loading {tool_name}: {str(e)}")]
    
    app.add_directive('tool-docstring', ToolDocstringDirective)
