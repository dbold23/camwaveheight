"""Satellite-derived shoreline (SDS) via CoastSat / Google Earth Engine.

Primary toolchain is CoastSat (S2 / L8 / L9, cloud-masked, sub-pixel, tidally
corrected). This module orchestrates it and reduces the shoreline vectors to a
per-transect cross-shore position time series:

    data/satellite/shoreline/shoreline_position.parquet   (UTC index)
        columns: transect_id, cross_shore_position_m

The `cross_shore_position_m` / `transect_id` contract is the hand-off to the cam
perception work: the cam's own shoreline segmentation can later be regressed
against this absolute SDS exactly as cam Hs is regressed against CDIP.

CoastSat + Earth Engine + GDAL are the heavy `sat-shoreline` extra and are
imported lazily; everything raises an actionable error if the deps/creds are
absent. Feasibility: 5-day revisit, cloud-limited, ~5–10 m accuracy — a slow
change monitor, not real-time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from camwaveheight.satellite.common import bbox_from_site
from camwaveheight.site import Site

log = logging.getLogger(__name__)


def _require_coastsat():
    try:
        import ee  # noqa: F401
        from coastsat import SDS_shoreline  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Satellite-derived shoreline needs the sat-shoreline extra:\n"
            '  pip install -e ".[sat-shoreline]"\n'
            "  pip install git+https://github.com/kvos/CoastSat\n"
            "then `earthengine authenticate` and set EE_PROJECT. GDAL wheels are "
            "finicky on macOS — consider conda-forge for geopandas/rioxarray."
        ) from exc


def _aoi_polygon(site: Site) -> list[list[float]]:
    """Return the AOI as a GeoJSON-style ring of [lon, lat] pairs.

    Uses `site.aoi.polygon_geojson` if it points at a readable polygon, else the
    bbox rectangle.
    """
    if site.aoi is not None and site.aoi.polygon_geojson:
        p = Path(site.aoi.polygon_geojson)
        if p.exists():
            gj = json.loads(p.read_text())
            geom = gj["features"][0]["geometry"] if gj.get("type") == "FeatureCollection" else gj.get("geometry", gj)
            return geom["coordinates"][0]
    lonmin, latmin, lonmax, latmax = bbox_from_site(site)
    return [[lonmin, latmin], [lonmax, latmin], [lonmax, latmax], [lonmin, latmax], [lonmin, latmin]]


def define_transects(site: Site, spacing_m: float = 100.0) -> dict[str, list[list[float]]]:
    """Cross-shore transects across the AOI, keyed by stable transect_id.

    Placeholder geometry derived from the AOI bbox: evenly spaced alongshore, each
    spanning the cross-shore extent. Replace with surveyed transects when available
    — the IDs are the join key shared with the cam perception work, so keep them
    stable once published.
    """
    lonmin, latmin, lonmax, latmax = bbox_from_site(site)
    # Treat longitude as alongshore for a roughly N–S coast; one transect per
    # `spacing_m` (~1.11e5 m per degree latitude at this latitude).
    span_deg = lonmax - lonmin
    n = max(1, int(round(span_deg * 111_000 / spacing_m)))
    transects: dict[str, list[list[float]]] = {}
    for i in range(n + 1):
        lon = lonmin + span_deg * i / max(1, n)
        transects[f"T{i:02d}"] = [[lon, latmin], [lon, latmax]]
    return transects


def extract_shoreline_timeseries(
    site: Site,
    start,
    end,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run CoastSat over [start, end] and reduce to per-transect cross-shore position.

    Returns a UTC-indexed DataFrame with `transect_id` and `cross_shore_position_m`
    and writes `shoreline_position.parquet`, `shorelines.geojson`, `transects.geojson`
    under `out_dir`. Raises with install/auth guidance until the heavy extra is
    present (P3).
    """
    if out_dir is None:
        out_dir = Path(site.satellite.cache_root) / "shoreline"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the AOI + transects now — these are cheap and useful for QGIS even
    # before CoastSat runs.
    _write_geojson(out_dir / "transects.geojson", define_transects(site), kind="LineString")

    _require_coastsat()  # raises until sat-shoreline is installed + EE authed
    raise NotImplementedError(
        "CoastSat extraction is the P3 deliverable. With the extra installed and "
        "Earth Engine authed, wire SDS_download.retrieve_images → "
        "SDS_shoreline.extract_shorelines → SDS_transects.compute_intersection over "
        "define_transects(site), then tidal_correction(), writing "
        f"{out_dir / 'shoreline_position.parquet'}."
    )


def tidal_correction(
    positions: pd.DataFrame,
    tide_series: pd.Series,
    beach_slope: float = 0.1,
) -> pd.DataFrame:
    """Project shoreline positions to a common datum using the Bruun-style rule
    `dx = tide / tan(beta)` (beach_slope = tan beta). Returns a corrected copy.
    """
    out = positions.copy()
    tide = tide_series.reindex(out.index, method="nearest")
    out["cross_shore_position_m"] = out["cross_shore_position_m"] - tide / beach_slope
    return out


def fetch_sdb(site: Site, start, end) -> pd.DataFrame:
    """Satellite-derived bathymetry stub (P3+). Limited by surf-zone turbidity."""
    raise NotImplementedError(
        "Satellite-derived bathymetry is a P3+ stub; nearshore turbidity makes the "
        "surf zone largely unretrievable from optical SDB."
    )


def _write_geojson(path: Path, named_geoms: dict[str, list], kind: str = "LineString") -> None:
    features = [
        {
            "type": "Feature",
            "properties": {"transect_id": name},
            "geometry": {"type": kind, "coordinates": coords},
        }
        for name, coords in named_geoms.items()
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
    log.info("wrote %s (%d features)", path, len(features))
