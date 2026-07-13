# Configuration file for the socialverse documentation (Sphinx + sphinx-book-theme).
# Mirrors the omicverse readthedocs setup: MyST-Markdown sources, the Jupyter-Book
# style sphinx_book_theme, copy buttons, and repository / download / fullscreen
# toolbar buttons.

# -- Project information ------------------------------------------------------
project = "socialverse"
copyright = "2025-2026, socialverse contributors"
author = "OmicVerse / socialverse contributors"
release = "0.7.2"
version = "0.7.2"

# -- General configuration ----------------------------------------------------
extensions = [
    "myst_nb",           # MyST-Markdown + Jupyter notebooks (.ipynb); includes myst_parser
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

# Notebooks are committed with their outputs; render as-is, never re-execute.
nb_execution_mode = "off"
nb_merge_streams = True

# MyST-Markdown features used across the tutorials.
myst_enable_extensions = [
    "colon_fence",     # ::: admonitions (never collide with ``` code fences)
    "deflist",
    "dollarmath",
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "substitution",
    "tasklist",
    "linkify",
]
myst_heading_anchors = 3
myst_url_schemes = ["http", "https", "mailto"]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # legacy dev docs that live in docs/ but are not part of the RTD site
    "CONTRACT_CARDS.md",
    "LANDSCAPE.md",
    "README-full.md",
    # the notebooks/ dir is symlinked in; its README is not a doc page
    "tutorials/notebooks/README.md",
    "tutorials/notebooks/**/*.py",
]

# GFM heading levels can jump; and notebook prose sometimes links to sibling
# notebooks by a stale filename — don't fail the build on either.
suppress_warnings = ["myst.header", "myst.xref_missing"]

# -- HTML output --------------------------------------------------------------
html_theme = "sphinx_book_theme"
html_title = "socialverse"
html_logo = "_static/socialverse_logo.png"
html_favicon = "_static/favicon.svg"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False

html_theme_options = {
    # repository toolbar (github / edit / issues) — like omicverse
    "repository_url": "https://github.com/omicverse/socialverse",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    # navigation / TOC behaviour
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "announcement": (
        "socialverse — the AI-era entry point for social science research, "
        "from data to paper."
    ),
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/omicverse/socialverse",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/socialverse/",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
}

# Sidebar: use sphinx-book-theme defaults (logo + search + nav), like omicverse.

# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
