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
    "sphinx_design"
]

# recognise both .md and .rst
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]


# html_theme = 'furo'
html_theme = 'sphinx_book_theme'
# html_theme       = "sphinx_rtd_theme"  # Commented out to use furo theme
html_static_path = ["_static"]



# html_theme_options = {
#     "logo": {
#         "text": "🪽AgentFly\n",
#         "image_light": "_static/logo-light.png",
#         "image_dark": "_static/logo-dark.png",
#     }
# }
html_theme_options = {
    # "path_to_docs": "docs",
    "repository_url": "https://github.com/executablebooks/sphinx-book-theme",
    "repository_branch": "master",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    },
    "use_edit_page_button": True,
    "use_source_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_sidenotes": True,
    "show_toc_level": 2,
    "show_navbar_depth": 2,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "globaltoc_collapse": False,
    "announcement": (
        "⚠️The latest release refactored our HTML, "
        "so double-check your custom CSS rules!⚠️"
    ),
    "logo": {
        "image_dark": "_static/logo-wide-dark.svg",
        "text": "🪽AgentFly Document",  # Uncomment to try text with logo
    },
    "icon_links": [
        {
            "name": "Paper",
            "url": "https://arxiv.org/pdf/2507.14897",
            "icon": "https://cdn.simpleicons.org/arxiv",
            "type": "url",
        },
        {
            "name": "WANDB",
            "url": "https://wandb.ai/AgentRL/Open",
            "icon": "https://cdn.simpleicons.org/weightsandbiases",
            "type": "url"
        },
        {
            "name": "HF",
            "url": "https://huggingface.co/collections/Agent-One/agentfly-6882061c6cf08537cb66c12b",
            "icon": "https://cdn.simpleicons.org/huggingface/FF9A00",
            "type": "url",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/Agent-One-Lab/AgentFly",
            "icon": "https://cdn.simpleicons.org/github",
            "type": "url",
        },
    ],
    # For testing
    # "use_fullscreen_button": False,
    # "home_page_in_toc": True,
    # "extra_footer": "<a href='https://google.com'>Test</a>",  # DEPRECATED KEY
    # "show_navbar_depth": 2,
    # Testing layout areas
    # "navbar_start": ["test.html"],
    # "navbar_center": ["test.html"],
    # "navbar_end": ["test.html"],
    # "navbar_persistent": ["test.html"],
    # "footer_start": ["test.html"],
    # "footer_end": ["test.html"]
}


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
