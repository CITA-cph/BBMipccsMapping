"""
app/layout.py — MYTIGATE Dash UI layout.

4-step flow:
  Step 1 — Explore environment (base map, env colour layers)
  Step 2 — Draw farm & compute (newshape polygon → M12 → M2/M3/M4)
  Step 3 — Visualise results (results colour layers, scenario toggle)
  Step 4 — Export

Single interactive Plotly Scattermap. Squarish map ratio via sidebar widths.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from config import HARVEST_MONTHS, PRIMARY_HARVEST

MONTH_LABELS_ALL = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                    7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
ALL_MONTH_OPTIONS     = [{"label": MONTH_LABELS_ALL[m], "value": m}
                          for m in range(1, 13)]
HARVEST_MONTH_OPTIONS = [{"label": MONTH_LABELS_ALL[m], "value": m}
                          for m in HARVEST_MONTHS]

ENV_LAYER_OPTIONS = [
    {"label": "— No color —",               "value": "none"},
    {"label": "Exclusion zones",            "value": "exclusions"},
    {"label": "Temperature  [°C]",          "value": "temp_mean"},
    {"label": "Salinity  [psu]",            "value": "sal_mean"},
    {"label": "Chlorophyll-a  [ln µg/L]",  "value": "lnchla_mean"},
    {"label": "Flow speed  [m/s]",          "value": "vh"},
    {"label": "Bathymetry  [m]",            "value": "bathymetry_m"},
]
RESULT_LAYER_OPTIONS = [
    {"label": "N-reduction  [tN/farm]",    "value": "N_red"},
    {"label": "Harvest WW  [t/farm]",      "value": "harvest_WW"},
    {"label": "Shell DW  [t/farm]",        "value": "shell_DW"},
    {"label": "Mussel biomass p50  [gDW]", "value": "mbio_p50"},
    {"label": "Food limitation ratio",     "value": "vh_ratio"},
    {"label": "Conflict score",            "value": "conflict_score"},
]
RDEP_OPTIONS  = [{"label": f"{v} m", "value": v} for v in [2,4,6,8]]
ILOOP_OPTIONS = [
    {"label": "0.7 m  (dense)",  "value": 0.7},
    {"label": "1.0 m",           "value": 1.0},
    {"label": "1.5 m",           "value": 1.5},
    {"label": "2.0 m  (sparse)", "value": 2.0},
]
SSP_OPTIONS = [
    {"label": "SSP1-2.6  (~+1.8 °C)", "value": "ssp1_2_6"},
    {"label": "SSP2-4.5  (~+2.7 °C)", "value": "ssp2_4_5"},
    {"label": "SSP3-7.0  (~+3.6 °C)", "value": "ssp3_7_0"},
    {"label": "SSP5-8.5  (~+4.4 °C)", "value": "ssp5_8_5"},
]
PERIOD_OPTIONS = [
    {"label": "Mid-century  2041–2060",    "value": "mid"},
    {"label": "End-of-century  2071–2100", "value": "end"},
]

CSS = """
* { box-sizing:border-box; }
body { margin:0; font-family:'Inter',system-ui,sans-serif;
       font-size:13px; background:#0f1117; color:#e0e0e0; }
.app-shell { display:flex; flex-direction:column; height:100vh; overflow:hidden; }

/* Topbar */
.topbar { display:flex; align-items:center; justify-content:space-between;
          padding:0 16px; height:46px; background:#141820;
          border-bottom:1px solid #2a2f3a; flex-shrink:0; }
.topbar-left  { display:flex; align-items:center; }
.topbar-icon  { font-size:1.3rem; margin-right:8px; }
.topbar-title { font-size:.95rem; font-weight:700; color:#f0f0f0; letter-spacing:.03em; }
.topbar-sub   { font-size:.68rem; color:#788; margin-left:10px; }
.topbar-status{ font-size:.72rem; color:#4caf79; }
.topbar-right { display:flex; align-items:center; gap:12px; }

/* Main */
.main-row { display:flex; flex:1; overflow:hidden; }

/* Sidebars — wide so map is squarish */
.sidebar { width:340px; flex-shrink:0; overflow-y:auto; background:#141820;
           border-right:1px solid #2a2f3a; padding:10px 16px 32px; }
.right-panel { width:320px; flex-shrink:0; overflow-y:auto; background:#141820;
               border-left:1px solid #2a2f3a; padding:10px 16px 32px; }

/* Step headers */
.step-header { display:flex; align-items:center; margin:16px 0 3px; }
.step-num  { width:20px; height:20px; border-radius:50%; background:#1e2b3a;
             color:#7ec8e3; font-size:.68rem; font-weight:700;
             display:flex; align-items:center; justify-content:center;
             flex-shrink:0; margin-right:7px; }
.step-num.active { background:#1a4a6a; color:#7ec8e3; }
.step-num.done   { background:#1a4a2a; color:#4caf79; }
.step-label { font-size:.72rem; font-weight:600; color:#aab;
              text-transform:uppercase; letter-spacing:.07em; }
.sidebar-hr { border:none; border-top:1px solid #2a2f3a; margin:0 0 7px; }
.sidebar-hint { font-size:.70rem; color:#566; margin:0 0 6px; line-height:1.4; }
.ctrl-label { font-size:.72rem; color:#aab; display:block; margin:6px 0 2px; }

/* Draw instruction */
.draw-box { background:#1a2535; border:1px solid #2a3f55; border-radius:5px;
            padding:8px 10px; margin:6px 0 8px; }
.draw-box p  { font-size:.72rem; color:#8ab; margin:0 0 3px; line-height:1.4; }
.draw-steps  { font-size:.68rem; color:#677; margin:0; padding-left:14px;
               line-height:1.7; }

/* Dropdowns */
.dd { background:#1c2230 !important; color:#e0e0e0 !important;
      font-size:.78rem !important; }
.Select-control,.Select-menu-outer { background:#1c2230 !important; color:#e0e0e0 !important; }
.Select-value-label { color:#e0e0e0 !important; }

/* Sliders */
.slider-wrap  { margin:6px 0 8px; }
.slider-label { font-size:.70rem; color:#dde; display:block; margin-bottom:3px; }
.rc-slider-rail        { background:#3a4a5a !important; height:4px !important; }
.rc-slider-track       { background:#4caf79 !important; height:4px !important; }
.rc-slider-handle      { border-color:#4caf79 !important; background:#ffffff !important;
                         width:14px !important; height:14px !important;
                         margin-top:-5px !important; }
.rc-slider-mark-text   { color:#ffffff !important; font-size:.68rem !important; }
.rc-slider-dot         { background:#3a4a5a !important; border-color:#4caf79 !important; }
.rc-slider-tooltip-inner { background:#1c2230 !important; color:#ffffff !important;
                            font-size:.68rem !important; border:1px solid #3a4a5a !important; }
.dash-slider .rc-slider-rail  { background:#3a4a5a !important; }
.dash-slider .rc-slider-track { background:#4caf79 !important; }
/* Force track via the Dash-rendered class wrapper */
.white-slider .rc-slider-track { background:#ffffff !important; height:4px !important; }
.white-slider .rc-slider-rail  { background:#3a4a5a !important; height:4px !important; }
.white-slider .rc-slider-handle { background:#ffffff !important;
                                   border-color:#ffffff !important; }
.white-slider .rc-slider-step   { background:transparent !important; }

/* Status */
.status-text { font-size:.70rem; color:#4caf79; margin-top:4px;
               min-height:14px; line-height:1.4; }

/* Map area */
.map-area { flex:1; overflow:hidden; position:relative; background:#0a0d14; }
.map-toggle { position:absolute; top:10px; left:50%;
              transform:translateX(-50%); z-index:20;
              display:flex; border-radius:6px; overflow:hidden;
              border:1px solid #2a2f3a; }
.map-toggle-btn { padding:4px 18px; font-size:.73rem; font-weight:600;
                  cursor:pointer; border:none; transition:background .15s; }
.map-toggle-btn.active   { background:#2a4a3a; color:#7effc0; }
.map-toggle-btn.inactive { background:#1c2230; color:#788; }

/* Right panel */
.panel-title { font-size:.62rem; font-weight:600; color:#566;
               text-transform:uppercase; letter-spacing:.08em; margin:14px 0 4px; }
.panel-hint  { font-size:.70rem; color:#455; line-height:1.4; }
.panel-section { margin-bottom:8px; }
.summary-card { background:#1a2535; border-radius:5px; padding:8px 10px; margin-bottom:6px; }
.summary-card-title { font-size:.60rem; color:#566; text-transform:uppercase;
                       letter-spacing:.07em; margin-bottom:4px; }
.metric-row   { display:flex; justify-content:space-between;
                padding:2px 0; border-bottom:1px solid #1c2230; }
.metric-label { color:#788; font-size:.70rem; }
.metric-value { color:#e0e0e0; font-weight:600; font-size:.70rem; }
.cell-row   { display:flex; justify-content:space-between; padding:2px 0;
              border-bottom:1px solid #1c2230; }
.cell-label { color:#788; font-size:.68rem; }
.cell-value { color:#d0d0d0; font-size:.68rem; font-weight:500; }
.cell-group-label { font-size:.60rem; color:#455; text-transform:uppercase;
                    letter-spacing:.06em; margin:6px 0 2px; }
.badge-row { margin-bottom:5px; }
.compare-table  { width:100%; }
.compare-header { display:grid; grid-template-columns:1fr 46px 46px 38px;
                  font-size:.60rem; color:#455; text-transform:uppercase; padding:2px 0; }
.compare-row    { display:grid; grid-template-columns:1fr 46px 46px 38px;
                  padding:2px 0; border-bottom:1px solid #1c2230; }
.compare-col-label { font-size:.68rem; color:#788; }
.compare-col-val   { font-size:.68rem; color:#d0d0d0; text-align:right; }
.compare-col-delta { font-size:.68rem; font-weight:600; text-align:right; }
.delta-pos { color:#4caf79; }
.delta-neg { color:#f06a6a; }
.export-btn { margin-bottom:5px !important; font-size:.73rem !important; }
"""


def _step(num, label, state=""):
    cls  = f"step-num {state}".strip()
    icon = "✓" if state == "done" else str(num)
    return html.Div(className="step-header", children=[
        html.Div(icon, className=cls),
        html.Span(label, className="step-label"),
    ])

def _hr():
    return html.Hr(className="sidebar-hr")

def _lbl(text):
    return html.Span(text, className="ctrl-label")

def _dd(sid, opts, val, **kw):
    return dcc.Dropdown(id=sid, options=opts, value=val,
                        clearable=False, searchable=False,
                        className="dd", **kw)

def _white_marks(marks: dict) -> dict:
    """Force white text on slider marks — CSS class overrides don't reach rc-slider."""
    return {k: {"label": str(v),
                "style": {"color": "#ffffff", "fontSize": "0.68rem"}}
            for k, v in marks.items()}

def _slider(sid, label, mn, mx, step, val, marks):
    return html.Div([
        html.Label(label, className="slider-label"),
        dcc.Slider(id=sid, min=mn, max=mx, step=step, value=val,
                   marks=_white_marks(marks),
                   tooltip={"placement": "bottom", "always_visible": False},
                   className="white-slider"),
    ], className="slider-wrap")


def build_layout():
    return html.Div(className="app-shell", children=[

        # Topbar
        html.Div(className="topbar", children=[
            html.Div(className="topbar-left", children=[
                html.Span("🦪", className="topbar-icon"),
                html.Span("MYTIGATE_futures", className="topbar-title"),
                html.Span("Mussel mitigation farm site selection · Baltic and Atlantic Denmark",
                          className="topbar-sub"),
            ]),
            html.Div(className="topbar-right", children=[
                html.Span(id="topbar-status", className="topbar-status"),
            ]),
        ]),

        html.Div(className="main-row", children=[

            # ── LEFT SIDEBAR ─────────────────────────────────────────────────
            html.Div(className="sidebar", children=[

                # Scenario mode — always at top
                _step(0, "Scenario mode", "active"),
                _hr(),
                dcc.RadioItems(
                    id="radio-mode",
                    options=[
                        {"label": " Baseline",             "value": "baseline"},
                        {"label": " CMIP6 scenario",       "value": "cmip6"},
                        {"label": " Custom delta  (slow)", "value": "delta"},
                    ],
                    value="baseline",
                    labelStyle={"display": "block", "marginBottom": "3px",
                                "fontSize": ".78rem", "color": "#bbb",
                                "cursor": "pointer"},
                ),
                html.Div(id="div-cmip6-panel", style={"display": "none"}, children=[
                    _lbl("SSP"),
                    _dd("dd-ssp", SSP_OPTIONS, "ssp2_4_5"),
                    _lbl("Period"),
                    _dd("dd-period", PERIOD_OPTIONS, "end"),
                    dbc.Button("Load scenario", id="btn-load-scenario",
                               color="info", size="sm", className="w-100 mt-2",
                               style={"fontSize": ".76rem"}),
                    dcc.Loading(
                        html.Div(id="div-scenario-status", className="status-text"),
                        type="dot", color="#4caf79",
                    ),
                ]),
                html.Div(id="div-delta-panel", style={"display": "none"}, children=[
                    _slider("slider-delta-temp", "ΔTemp [°C]",
                            -2, 6, 0.1, 0.0, {-2:"-2", 0:"0", 3:"+3", 6:"+6"}),
                    _slider("slider-delta-sal", "ΔSal [psu]",
                            -3, 1, 0.1, 0.0, {-3:"-3", 0:"0", 1:"+1"}),
                    _slider("slider-chla-mult", "ChlA ×",
                            0.5, 2.0, 0.05, 1.0, {0.5:"×0.5", 1:"×1", 2:"×2"}),
                    dbc.Button("Run scenario  (slow)", id="btn-run-delta",
                               color="warning", size="sm", className="w-100 mt-2",
                               style={"fontSize": ".76rem"}),
                    dcc.Loading(
                        html.Div(id="div-delta-status", className="status-text"),
                        type="dot", color="#f0a040",
                    ),
                ]),

                # Step 1
                _step(1, "Explore environment"),
                _hr(),
                html.P("Select a layer to colour the map. No model results yet.",
                       className="sidebar-hint"),
                _lbl("Colour layer"),
                _dd("dd-env-layer", ENV_LAYER_OPTIONS, "none"),
                _lbl("Month"),
                _dd("dd-env-month", ALL_MONTH_OPTIONS, 7),
                _slider("slider-marker-size", "Marker size",
                        2, 12, 1, 6,
                        {2:"2", 4:"4", 6:"6", 8:"8", 10:"10", 12:"12"}),

                # Step 2
                _step(2, "Draw farm & compute"),
                _hr(),
                html.Div(className="draw-box", children=[
                    html.P("Use the draw tool in the map toolbar:"),
                    html.Ol(className="draw-steps", children=[
                        html.Li("Zoom in to your site of interest (zoom ≥ 10)"),
                        html.Li("Click the heart/lasso icon (♡) in the map toolbar"),
                        html.Li("Click and drag to draw the farm polygon"),
                        html.Li("Release to close the polygon"),
                        html.Li("Press Compute below"),
                    ]),
                    html.Div(id="div-zoom-warning",
                             style={"marginTop": "6px", "fontSize": ".70rem",
                                    "fontWeight": "600"}),
                ]),
                _slider("slider-max-rdep", "Max collector depth [m]",
                        2, 8, None, 2, {2:"2m", 4:"4m", 6:"6m", 8:"8m"}),
                _slider("slider-iloop", "Loop interval [m]",
                        0, 3, None, 0, {0:"0.7m", 1:"1.0m", 2:"1.5m", 3:"2.0m"}),
                html.Div(id="div-restriction-warning",
                         style={"fontSize": ".70rem", "fontWeight": "600",
                                "marginBottom": "6px", "minHeight": "14px"}),
                dbc.Button("⚙  Compute farm", id="btn-compute",
                           color="primary", className="w-100 mt-2",
                           style={"fontSize": ".80rem"}),
                dcc.Loading(
                    html.Div(id="div-compute-status", className="status-text"),
                    type="dot", color="#4caf79",
                ),

                # Step 3
                _step(3, "Visualise results"),
                _hr(),
                _lbl("Result layer"),
                _dd("dd-result-layer", RESULT_LAYER_OPTIONS, "N_red", disabled=True),
                _lbl("Harvest month"),
                _dd("dd-harvest-month", HARVEST_MONTH_OPTIONS, PRIMARY_HARVEST,
                    disabled=True),

                # Step 4
                _step(4, "Export"),
                _hr(),
                html.P(id="export-hint",
                       children="Available after Step 2 completes.",
                       className="sidebar-hint"),
                dbc.Button("⬇  Env grid — baseline", id="btn-dl-env-baseline",
                           color="secondary", outline=True, disabled=True,
                           className="w-100 export-btn"),
                dbc.Button("⬇  Results — baseline", id="btn-dl-results-baseline",
                           color="secondary", outline=True, disabled=True,
                           className="w-100 export-btn"),
                html.Div(id="div-export-scenario", style={"display":"none"}, children=[
                    dbc.Button("⬇  Env grid — scenario", id="btn-dl-env-scenario",
                               color="success", outline=True, disabled=True,
                               className="w-100 export-btn"),
                    dbc.Button("⬇  Results — scenario", id="btn-dl-results-scenario",
                               color="success", outline=True, disabled=True,
                               className="w-100 export-btn"),
                ]),
                dbc.Button("⬇  Farm summary (CSV)", id="btn-dl-farm-summary",
                           color="secondary", outline=True, disabled=True,
                           className="w-100 export-btn"),
                dbc.Button("⬇  Farm cell summary (CSV)", id="btn-dl-cell-summary",
                           color="secondary", outline=True, disabled=True,
                           className="w-100 export-btn"),
                dcc.Download(id="dl-env-baseline"),
                dcc.Download(id="dl-results-baseline"),
                dcc.Download(id="dl-env-scenario"),
                dcc.Download(id="dl-results-scenario"),
                dcc.Download(id="dl-farm-summary"),
                dcc.Download(id="dl-cell-summary"),
            ]),

            # ── MAP ───────────────────────────────────────────────────────────
            html.Div(className="map-area", children=[
                # Baseline / Scenario toggle
                html.Div(id="div-map-toggle", className="map-toggle",
                         style={"display":"none"}, children=[
                    html.Button("Baseline", id="btn-view-baseline",
                                className="map-toggle-btn active", n_clicks=0),
                    html.Button("Scenario", id="btn-view-scenario",
                                className="map-toggle-btn inactive", n_clicks=0),
                ]),
                # Crop toggle — shown after compute
                html.Div(id="div-crop-toggle", className="map-toggle",
                         style={"display":"none",
                                "top":"10px", "left":"calc(50% + 160px)",
                                "transform":"none"}, children=[
                    html.Button("Farm only", id="btn-crop-farm",
                                className="map-toggle-btn active", n_clicks=0),
                    html.Button("Full grid", id="btn-crop-full",
                                className="map-toggle-btn inactive", n_clicks=0),
                ]),
                dcc.Graph(
                    id="map-graph",
                    style={"height":"100%","width":"100%"},
                    config={
                        "scrollZoom": True,
                        "modeBarButtonsToRemove": ["lasso2d","select2d"],
                        "modeBarButtonsToAdd":    ["drawclosedpath","eraseshape"],
                        "displaylogo": False,
                        "editable": True,
                        "edits": {"shapePosition": False},
                    },
                ),
            ]),

            # ── RIGHT PANEL ───────────────────────────────────────────────────
            html.Div(className="right-panel", children=[
                html.Div(className="panel-section", children=[
                    html.P("Farm summary", className="panel-title"),
                    html.Div(id="div-farm-summary",
                             children=html.Span(
                                 "Draw a farm polygon and press Compute.",
                                 className="panel-hint")),
                ]),
                html.Div(className="panel-section", children=[
                    html.P("Clicked cell", className="panel-title"),
                    html.Div(id="div-cell-report",
                             children=html.Span("Click any cell on the map.",
                                                className="panel-hint")),
                ]),
                html.Div(id="div-compare-panel", className="panel-section",
                         style={"display":"none"}, children=[
                    html.P("Baseline vs scenario", className="panel-title"),
                    html.Div(id="div-compare-table"),
                ]),
            ]),
        ]),

        # Hidden stores
        dcc.Store(id="store-compute-done",   data=False),
        dcc.Store(id="store-crop-mode",      data="farm"),  # "farm" | "full"
        dcc.Store(id="store-scenario-ready", data=False),
        dcc.Store(id="store-active-view",    data="baseline"),
        dcc.Store(id="store-scenario-label", data=""),
        dcc.Store(id="store-farm-cells-ids"),
        dcc.Store(id="store-drawn-shape"),   # holds {"shape_path": "..."}
        dcc.Store(id="store-map-viewport",
                  data={"center": {"lon": 11.5, "lat": 56.3}, "zoom": 7.0}),
    ])


# ── Render helpers ─────────────────────────────────────────────────────────────

def render_farm_summary(m12: dict, m_str: str) -> html.Div:
    farm_cells = m12.get("farm_cells")
    if farm_cells is None or farm_cells.empty:
        return html.Span("No viable cells in polygon.", className="panel-hint")

    n_cells  = len(farm_cells)
    n_viable = int(farm_cells["viable"].sum()) \
               if "viable" in farm_cells.columns else n_cells

    def _sum(col):
        v = farm_cells[col].fillna(0).sum() if col in farm_cells.columns else None
        return float(v) if v is not None else None

    def _mean_pos(col):
        if col not in farm_cells.columns:
            return None
        v = farm_cells[col].fillna(0)
        pos = v[v > 0]
        return float(pos.mean()) if len(pos) else None

    def row(label, val, unit="", fmt=".2f"):
        if val is None:
            return None
        txt = f"{val:{fmt}} {unit}".strip()
        return html.Div(className="metric-row", children=[
            html.Span(label, className="metric-label"),
            html.Span(txt,   className="metric-value"),
        ])

    geom = [
        row("Farm area",   m12.get("farm_area_m2", 0)/1e4, "ha",  ".1f"),
        row("Sections",    m12.get("n_sections"),           "",    "d"),
        row("D_short",     m12.get("D_short"),              "m",   ".0f"),
        row("D_long",      m12.get("D_long"),               "m",   ".0f"),
        row("l_col mean",  m12.get("l_col"),                "m",   ".0f"),
        row("Orientation", m12.get("orientation_deg"),      "°",   ".1f"),
        row("Flow mean",   m12.get("vh_mean"),              "m/s", ".3f"),
        row("Viable cells",n_viable, f"/ {n_cells}",              "d"),
    ]
    results = [
        row("N-reduction", _sum(f"N_red_{m_str}"),       "tN"),
        row("P-reduction", _sum(f"P_red_{m_str}"),       "tP",  ".3f"),
        row("Harvest WW",  _sum(f"harvest_WW_{m_str}"),  "t"),
        row("Shell DW",    _sum(f"shell_DW_{m_str}"),    "t"),
        row("mbio p50",    _mean_pos(f"mbio_p50_{m_str}"), "gDW", ".4f"),
        row("vh_ratio",    _mean_pos(f"vh_ratio_{m_str}"), "",   ".3f"),
    ]
    return html.Div([
        html.Div(className="summary-card", children=[
            html.Div("Farm geometry", className="summary-card-title"),
            *[r for r in geom if r],
        ]),
        html.Div(className="summary-card", children=[
            html.Div(f"Results — month {m_str}", className="summary-card-title"),
            *[r for r in results if r],
        ]),
    ])


def render_cell_report(report: dict) -> html.Div:
    if not report or "error" in report:
        return dbc.Alert(report.get("error", "No data"), color="danger")

    def row(label, val, unit=""):
        if val is None:
            return None
        txt = (f"{val:.3f} {unit}".strip()
               if isinstance(val, float) else f"{val} {unit}".strip())
        return html.Div(className="cell-row", children=[
            html.Span(label, className="cell-label"),
            html.Span(txt,   className="cell-value"),
        ])

    excl    = report.get("excluded", False)
    food_ok = report.get("food_ok", True)
    return html.Div([s for s in [
        html.Div(className="badge-row", children=[
            dbc.Badge("EXCLUDED" if excl else "Available",
                      color="danger" if excl else "success"),
            dbc.Badge("Food OK" if food_ok else "Food risk",
                      color="success" if food_ok else "warning", className="ms-1"),
        ]),
        html.Div(className="cell-group-label", children="Location"),
        row("Depth",   report.get("bathymetry_m"), "m"),
        row("rdep",    report.get("rdep_m"),        "m"),
        row("Conflict",report.get("conflict_score")),
        row("vh_ratio",report.get("vh_ratio")),
        html.Div(className="cell-group-label", children="Mussel"),
        row("mbio p05", report.get("mbio_p05_gDW"), "gDW"),
        row("mbio p50", report.get("mbio_p50_gDW"), "gDW"),
        row("mbio p95", report.get("mbio_p95_gDW"), "gDW"),
        html.Div(className="cell-group-label", children="Farm"),
        row("N-red",   report.get("N_red_tN"),     "tN"),
        row("P-red",   report.get("P_red_tP"),     "tP"),
        row("Harvest", report.get("harvest_WW_t"), "t"),
        row("N/ha",    report.get("N_red_ha"),     "tN/ha"),
        html.Div(className="cell-group-label", children="Environment"),
        row("Temp",    report.get("temp_mean_C"),  "°C"),
        row("Sal",     report.get("sal_mean_psu"), "psu"),
        row("ChlA",    report.get("chla_ugL"),     "µg/L"),
    ] if s is not None])


def render_compare_table(rep_base: dict, rep_scen: dict, label: str) -> html.Div:
    if not rep_base or not rep_scen:
        return html.Span("Click a cell.", className="panel-hint")
    keys = [
        ("N-red tN",  "N_red_tN"),
        ("P-red tP",  "P_red_tP"),
        ("Harvest t", "harvest_WW_t"),
        ("mbio p50",  "mbio_p50_gDW"),
        ("Temp °C",   "temp_mean_C"),
        ("Sal psu",   "sal_mean_psu"),
        ("ChlA",      "chla_ugL"),
        ("vh_ratio",  "vh_ratio"),
    ]
    short = label[:10]
    rows = [html.Div(className="compare-header", children=[
        html.Span("",     className="compare-col-label"),
        html.Span("Base", className="compare-col-val"),
        html.Span(short,  className="compare-col-val"),
        html.Span("Δ",    className="compare-col-delta"),
    ])]
    for lbl, key in keys:
        bv = rep_base.get(key) or 0.0
        sv = rep_scen.get(key) or 0.0
        d  = sv - bv
        rows.append(html.Div(className="compare-row", children=[
            html.Span(lbl,          className="compare-col-label"),
            html.Span(f"{bv:.2f}", className="compare-col-val"),
            html.Span(f"{sv:.2f}", className="compare-col-val"),
            html.Span(f"{d:+.2f}",
                      className=f"compare-col-delta {'delta-pos' if d>=0 else 'delta-neg'}"),
        ]))
    return html.Div(rows, className="compare-table")