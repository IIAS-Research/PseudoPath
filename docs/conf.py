project = "PseudoPath"
author = "IIAS Research"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "_templates", "_static"]
myst_heading_anchors = 4
html_theme = "furo"
html_title = "PseudoPath"
html_baseurl = "https://iias-research.github.io/PseudoPath/"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_css_files = ["pseudopath.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#4b46a6",
        "color-brand-content": "#5146a5",
        "color-foreground-primary": "#25263c",
        "color-background-secondary": "#f6f6fb",
        "color-background-border": "#e1e2ee",
    },
    "dark_css_variables": {
        "color-brand-primary": "#b5adff",
        "color-brand-content": "#c4beff",
        "color-foreground-primary": "#e9e7f5",
        "color-background-primary": "#181725",
        "color-background-secondary": "#211f32",
        "color-background-border": "#3c3854",
    },
}


def add_homepage_assets(app, pagename, _templatename, _context, _doctree):
    if pagename == "index":
        app.add_css_file("benchmark.css")
        app.add_js_file("benchmark.js", defer="defer")


def setup(app):
    app.connect("html-page-context", add_homepage_assets)
