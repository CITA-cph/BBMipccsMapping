"""
app/app.py — MYTIGATE entry point.
Run: python app/app.py → http://localhost:8050
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dash
import dash_bootstrap_components as dbc

from app.layout import build_layout, CSS
from app.callbacks import register_callbacks, load_all_grids


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="MYTIGATE_futures",
        suppress_callback_exceptions=True,
        meta_tags=[{"name": "viewport",
                    "content": "width=device-width, initial-scale=1"}],
    )
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{CSS}</style></head>"
    )
    app.layout = build_layout()
    load_all_grids(app)
    register_callbacks(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8050, use_reloader=False)