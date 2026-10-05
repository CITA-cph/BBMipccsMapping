"""
app/map_utils.py
----------------
Plotly Scattermapbox map builder.

Token: set MAPBOX_TOKEN in config.py.
The token is passed via mapbox_accesstoken in update_layout —
this is the correct Plotly API (underscore-separated, not dot-nested).
"""
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go


# MAP_CENTER and MAP_ZOOM are computed dynamically from grid data in build_map().
# These are fallback defaults only (used by empty_map and before grid is loaded).
_DEFAULT_CENTER = dict(lat=56.3, lon=10.4)
_DEFAULT_ZOOM   = 7.0
MAP_STYLE       = "carto-positron"   # Plotly v6 built-in: carto-positron, carto-darkmatter, open-street-map, white-bg


def _compute_center_zoom(lats: np.ndarray, lons: np.ndarray) -> tuple[dict, float]:
    """
    Derive map center and zoom from the bounding box of the grid data.
    Zoom is estimated from the latitude span using the standard Web Mercator
    tile formula (approximate but sufficient for fixed-aspect Plotly maps).
    """
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    center = dict(lat=float((lat_min + lat_max) / 2),
                  lon=float((lon_min + lon_max) / 2))
    # Approximate zoom: ln2(360 / lon_span) capped to reasonable range
    lon_span = max(lon_max - lon_min, 0.1)
    zoom = float(np.clip(np.log2(360.0 / lon_span) - 1, 4.0, 12.0))
    return center, zoom

COLOURSCALES = {
    # Result layers
    "N_red":         "YlOrRd",
    "harvest_WW":    "Greens",
    "harvest_DW":    "Greens",
    "shell_DW":      "Greens",
    "mbio_p50":      "Blues",
    "vh_ratio":      "RdYlGn",
    "conflict_score":"Reds",
    # Env layers
    "temp_mean":     "RdYlBu_r",
    "sal_mean":      "BrBG",
    "lnchla_mean":   "YlGn",
    "vh":            "Blues",
    "bathymetry_m":  "deep",
}

ENV_COLS_NO_MONTH = {"bathymetry_m", "conflict_score"}


def _mapbox_layout(title: str = "",
                   center: dict = None,
                   zoom: float = None,
                   preserve_viewport: bool = False) -> dict:
    """
    Shared map layout config.

    preserve_viewport=True: do not set center/zoom — Plotly keeps whatever
    the user has panned/zoomed to. Use this for all re-renders after the
    initial load so the map never jumps.

    uirevision="static" ensures Plotly never resets the viewport on any
    figure update (layer change, result update, etc.).
    """
    map_cfg = {"style": MAP_STYLE}
    if not preserve_viewport:
        map_cfg["center"] = center or _DEFAULT_CENTER
        map_cfg["zoom"]   = zoom   or _DEFAULT_ZOOM

    return dict(
        map=map_cfg,
        margin=dict(l=0, r=0, t=24 if title else 0, b=0),
        title=dict(text=title, x=0.5, font=dict(size=13)) if title else None,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    borderwidth=1, bordercolor="#ccc"),
        uirevision="static",   # never reset viewport on figure update
    )


def build_map(
    gdf: gpd.GeoDataFrame,
    colour_col: str,
    harvest_month: int,
    selected_ids: list[int] | None = None,
    title: str = "",
) -> go.Figure:
    m   = f"{harvest_month:02d}"
    col = (f"{colour_col}_{m}"
           if colour_col not in ("conflict_score", "suitability",
                                  "food_limitation_severity")
           else colour_col)
    if col not in gdf.columns:
        candidates = [c for c in gdf.columns if colour_col in c]
        col = candidates[0] if candidates else None

    values = gdf[col].fillna(0).values.astype(float) if col else np.zeros(len(gdf))
    pos    = values[values > 0]
    vmax   = float(np.percentile(pos, 98)) if len(pos) else 1.0
    cscale = COLOURSCALES.get(colour_col, "Viridis")

    n_col  = f"N_red_{m}"
    b_col  = "bathymetry_m"
    n_vals = gdf[n_col].fillna(0).values if n_col in gdf.columns else np.zeros(len(gdf))
    b_vals = gdf[b_col].fillna(0).values if b_col in gdf.columns else np.zeros(len(gdf))

    hover = [
        f"Cell {cid}<br>N-red: {nr:.2f} tN<br>Depth: {bd:.1f} m"
        for cid, nr, bd in zip(gdf["cell_id"].values, n_vals, b_vals)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=gdf["lat"].values,
        lon=gdf["lon"].values,
        mode="markers",
        marker=dict(
            size=6, color=values, colorscale=cscale,
            cmin=0, cmax=vmax, opacity=0.75,
            colorbar=dict(title=dict(text=col, side="right"),
                          thickness=10, len=0.55, x=1.0),
        ),
        text=hover, hoverinfo="text",
        customdata=gdf["cell_id"].values,
        name="Cells",
    ))

    if selected_ids:
        sel = gdf[gdf["cell_id"].isin(selected_ids)]
        if not sel.empty:
            fig.add_trace(go.Scattermap(
                lat=sel["lat"].values, lon=sel["lon"].values,
                mode="markers",
                marker=dict(size=8, color="cyan", opacity=1.0),
                name="Selected",
                hoverinfo="text",
                text=[f"SELECTED — Cell {c}" for c in sel["cell_id"].values],
            ))

    fig.update_layout(**_mapbox_layout(title, preserve_viewport=True))
    return fig


def build_env_map(
    gdf: gpd.GeoDataFrame,
    env_col: str,
    harvest_month: int,
    title: str = "",
) -> go.Figure:
    """
    Step 1 map — shows a single env variable across the full grid.
    No M2/M3/M4 columns needed. Used before farm compute.
    """
    m = f"{harvest_month:02d}"

    # Build full column name
    if env_col in ENV_COLS_NO_MONTH:
        col = env_col
    else:
        col = f"{env_col}_{m}"
        if col not in gdf.columns:
            # fallback: any month
            candidates = [c for c in gdf.columns if c.startswith(env_col)]
            col = candidates[0] if candidates else None

    values = gdf[col].fillna(np.nan).values.astype(float) if col else np.zeros(len(gdf))
    finite = values[np.isfinite(values)]
    vmin   = float(np.percentile(finite, 2))  if len(finite) else 0.0
    vmax   = float(np.percentile(finite, 98)) if len(finite) else 1.0
    cscale = COLOURSCALES.get(env_col, "Viridis")

    hover = [
        f"Cell {cid}<br>{col}: {v:.3f}"
        for cid, v in zip(gdf["cell_id"].values, values)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=gdf["lat"].values,
        lon=gdf["lon"].values,
        mode="markers",
        marker=dict(
            size=6, color=values, colorscale=cscale,
            cmin=vmin, cmax=vmax, opacity=0.80,
            colorbar=dict(title=dict(text=col, side="right"),
                          thickness=10, len=0.55, x=1.0),
        ),
        text=hover, hoverinfo="text",
        customdata=gdf["cell_id"].values,
        name=col or env_col,
    ))
    fig.update_layout(**_mapbox_layout(title or col or env_col,
                                       preserve_viewport=True))
    return fig


def empty_map(message: str = "No data — run pipeline.py first") -> go.Figure:
    """Initial/error map — sets the starting viewport. All subsequent renders preserve it."""
    center, zoom = _compute_center_zoom(
        np.array([54.6, 58.0]), np.array([7.8, 13.1])
    )
    fig = go.Figure(go.Scattermap())
    fig.update_layout(**_mapbox_layout(preserve_viewport=False,
                                       center=center, zoom=zoom))
    fig.update_layout(
        title=dict(text=f"⚠  {message}", x=0.5) if message else None,
        uirevision="static",
    )
    return fig