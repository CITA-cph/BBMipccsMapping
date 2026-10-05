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
    shapes: list | None = None,
    farm_gdf: gpd.GeoDataFrame | None = None,
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

    b_col  = "bathymetry_m"
    b_vals = gdf[b_col].fillna(0).values if b_col in gdf.columns else np.zeros(len(gdf))
    lats_h = gdf["lat"].values
    lons_h = gdf["lon"].values

    # Show current layer value in hover, not hardcoded N_red
    layer_label = col or colour_col
    hover = [
        f"Lat: {la:.4f}  Lon: {lo:.4f}<br>Depth: {bd:.1f} m<br>{layer_label}: {v:.3f}"
        for la, lo, bd, v in zip(lats_h, lons_h, b_vals, values)
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

    # Farm boundary — geo-anchored outline trace (scales with zoom)
    if farm_gdf is not None:
        boundary = farm_boundary_trace(farm_gdf)
        if boundary is not None:
            fig.add_trace(boundary)

    layout = _mapbox_layout(title, preserve_viewport=True)
    if shapes:
        layout["shapes"] = shapes
    fig.update_layout(**layout)
    return fig


def build_exclusion_outline_map(
    gdf: gpd.GeoDataFrame,
    shapes: list | None = None,
) -> go.Figure:
    """
    Startup map: plain basemap with exclusion zone outlines.
    Excluded cells shown as red outline dots; available cells as grey dots.
    No color fill gradient — just spatial context of where students can/cannot draw.
    """
    center, zoom = _compute_center_zoom(gdf["lat"].values, gdf["lon"].values)

    excl    = gdf["excluded"].fillna(False).values if "excluded" in gdf.columns \
              else np.zeros(len(gdf), dtype=bool)
    lats    = gdf["lat"].values
    lons    = gdf["lon"].values

    fig = go.Figure()

    # Available cells — subtle grey
    avail_mask = ~excl
    if avail_mask.any():
        fig.add_trace(go.Scattermap(
            lat=lats[avail_mask], lon=lons[avail_mask],
            mode="markers",
            marker=dict(size=4, color="#2a3a4a", opacity=0.5),
            hoverinfo="skip", name="Available",
            customdata=gdf["cell_id"].values[avail_mask],
        ))

    # Excluded cells — red outline
    if excl.any():
        fig.add_trace(go.Scattermap(
            lat=lats[excl], lon=lons[excl],
            mode="markers",
            marker=dict(size=5, color="#f06a6a", opacity=0.7),
            text=[f"Cell {c} — EXCLUDED"
                  for c in gdf["cell_id"].values[excl]],
            hoverinfo="text", name="Excluded",
            customdata=gdf["cell_id"].values[excl],
        ))

    layout = _mapbox_layout(
        "Exclusion zones — red = restricted, draw only in grey areas",
        preserve_viewport=False, center=center, zoom=zoom,
    )
    if shapes:
        layout["shapes"] = shapes
    fig.update_layout(**layout)
    return fig


def farm_boundary_trace(
    farm_gdf: gpd.GeoDataFrame,
    color: str = "cyan",
    width: int = 2,
) -> go.Scattermap | None:
    """
    Build a geo-anchored closed line trace outlining the farm cell boundaries.
    Uses the union of farm cell geometries projected to WGS84.
    Scales correctly with map zoom (geo-anchored, not paper-anchored).
    """
    if farm_gdf is None or farm_gdf.empty:
        return None
    try:
        farm_wgs  = farm_gdf.to_crs("EPSG:4326")
        union     = farm_wgs.geometry.union_all()
        # Extract exterior coords from union (Polygon or MultiPolygon)
        from shapely.geometry import MultiPolygon
        polys = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
        lats, lons = [], []
        for poly in polys:
            coords = list(poly.exterior.coords)
            lons  += [c[0] for c in coords] + [None]
            lats  += [c[1] for c in coords] + [None]
        return go.Scattermap(
            lat=lats, lon=lons,
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip",
            name="Farm boundary",
            showlegend=False,
        )
    except Exception as e:
        print(f"[Farm boundary] Error: {e}")
        return None


def build_env_map(
    gdf: gpd.GeoDataFrame,
    env_col: str,
    harvest_month: int,
    title: str = "",
    shapes: list | None = None,
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
    layout = _mapbox_layout(title or col or env_col, preserve_viewport=True)
    if shapes:
        layout["shapes"] = shapes
    fig.update_layout(**layout)
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