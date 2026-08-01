"""CMEMS satellite + model wave products — a `cdip.py` twin.

Three sources, all subset to the site bbox/time window and cached as parquet
(+ the raw `.nc`) exactly the way `cdip.fetch_cdip_params` caches CSV->parquet:

  fetch_cmems_wave_model       global analysis-forecast (default ~0.083°/3-hourly
                               NRT); VHM0/VTPK/VMDR -> model_hs/model_tp/model_dp
  fetch_cmems_wave_reanalysis  same vars from the multi-year WAVERYS reanalysis id
  fetch_altimeter_swh          L3 multi-mission along-track SWH, haversine-filtered
                               to <= altimeter_max_dist_km of the site

`copernicusmarine` is a heavy optional dep (the `sat-waves` extra) and is
imported lazily inside each fetch, so importing this module stays cheap.

Feasibility notes carried from the plan:
  - Model is a cross-check / forecast context source, not surf-zone truth: the
    global grid cell is offshore of the break.
  - Altimeter revisit is sparse and tracks are land-contaminated within ~10-20 km
    of the coast, so narrow windows routinely return zero rows. Use it only as an
    occasional independent cross-check.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from camwaveheight.satellite.common import (
    bbox_from_site,
    cache_key,
    coerce_utc_index,
    haversine_km,
    iso_utc,
    load_from_cache_or,
    site_lat_lon,
)
from camwaveheight.site import Site

log = logging.getLogger(__name__)

# CMEMS standard names -> our columns.
WAVE_VAR_MAP = {"VHM0": "model_hs", "VTPK": "model_tp", "VMDR": "model_dp"}
DEFAULT_WAVE_VARS = ("VHM0", "VTPK", "VMDR")
# L3 SWH variable names differ slightly across CMEMS products/missions.
ALT_SWH_CANDIDATES = ("VAVH", "VAVH_UNFILTERED", "VAVH_OCOG", "swh")


def _require_copernicusmarine():
    """Import copernicusmarine or raise an install/login hint (matches CLI msgs)."""
    try:
        import copernicusmarine  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "copernicusmarine is required for satellite wave products. Install it "
            'with:  pip install -e ".[sat-waves]"   then run `copernicusmarine login` '
            "once (free account: https://data.marine.copernicus.eu/register)."
        ) from e
    return copernicusmarine


def _coord_name(ds, *candidates):
    for c in candidates:
        if c in ds.coords or c in getattr(ds, "dims", ()):
            return c
    return None


def _nearest_point(ds, lat: float, lon: float):
    """Select the grid cell nearest (lat, lon), tolerating lat/latitude naming."""
    lat_name = _coord_name(ds, "latitude", "lat", "nav_lat")
    lon_name = _coord_name(ds, "longitude", "lon", "nav_lon")
    if lat_name and lon_name:
        return ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    return ds


def _subset_nc(cm, *, dataset_id, variables, bbox, start, end, nc_path: Path) -> None:
    """Run copernicusmarine.subset into `nc_path`. Kwargs follow the v1.x API."""
    lonmin, latmin, lonmax, latmax = bbox
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "copernicusmarine.subset %s vars=%s bbox=%s [%s..%s]",
        dataset_id, list(variables), bbox, iso_utc(start), iso_utc(end),
    )
    cm.subset(
        dataset_id=dataset_id,
        variables=list(variables),
        minimum_longitude=lonmin,
        maximum_longitude=lonmax,
        minimum_latitude=latmin,
        maximum_latitude=latmax,
        start_datetime=iso_utc(start),
        end_datetime=iso_utc(end),
        output_filename=nc_path.name,
        output_directory=str(nc_path.parent),
        overwrite=True,
        disable_progress_bar=True,
    )


def _model_df_from_nc(nc_path: Path, lat: float, lon: float, variables) -> pd.DataFrame:
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        pt = _nearest_point(ds, lat, lon)
        cols: dict[str, pd.Series] = {}
        for v in variables:
            if v in pt.variables:
                cols[WAVE_VAR_MAP.get(v, v)] = pt[v].to_series()
    if not cols:
        return _empty_model()
    df = pd.DataFrame(cols)
    df.index.name = "time"
    return coerce_utc_index(df)


def _empty_model() -> pd.DataFrame:
    return pd.DataFrame(columns=["model_hs", "model_tp", "model_dp"]).rename_axis("time")


def _empty_altimeter() -> pd.DataFrame:
    return pd.DataFrame(columns=["alt_swh", "alt_dist_km", "alt_mission"]).rename_axis("time")


def fetch_cmems_wave_model(
    site: Site,
    start,
    end,
    product_id: str | None = None,
    variables: tuple[str, ...] = DEFAULT_WAVE_VARS,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Nearest-grid-cell wave model time series for the site over [start, end] UTC.

    Returns a UTC-indexed DataFrame with `model_hs` (m), `model_tp` (s), `model_dp`
    (deg). Caches `<cache_dir>/cmemswave_<product_id>_<start>_<end>.parquet` plus the
    raw `.nc`. Empty frame if the subset yields no overlapping data.
    """
    product_id = product_id or site.satellite.products.cmems_wave_model_id
    if cache_dir is None:
        cache_dir = Path(site.satellite.cache_root) / "waves"
    key = cache_key(f"cmemswave_{product_id}", start, end)
    lat, lon = site_lat_lon(site)

    def _fetch() -> pd.DataFrame:
        cm = _require_copernicusmarine()
        nc_path = Path(cache_dir) / key.replace(".parquet", ".nc")
        _subset_nc(
            cm,
            dataset_id=product_id,
            variables=variables,
            bbox=bbox_from_site(site),
            start=start,
            end=end,
            nc_path=nc_path,
        )
        return _model_df_from_nc(nc_path, lat, lon, variables)

    return load_from_cache_or(cache_dir, key, _fetch)


def fetch_cmems_wave_reanalysis(
    site: Site, start, end, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """WAVERYS-class multi-year reanalysis (coarser grid, ~months latency).

    Thin wrapper over `fetch_cmems_wave_model` using the reanalysis product id; the
    distinct id keeps its cache file separate.
    """
    return fetch_cmems_wave_model(
        site,
        start,
        end,
        product_id=site.satellite.products.cmems_wave_reanalysis_id,
        cache_dir=cache_dir,
    )


def fetch_altimeter_swh(
    site: Site,
    start,
    end,
    product_id: str | None = None,
    max_dist_km: float | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """L3 multi-mission along-track SWH within `max_dist_km` of the site.

    Returns a UTC-indexed DataFrame with `alt_swh` (m), `alt_dist_km`, `alt_mission`.
    Expect empty windows: revisit is sparse and near-coast tracks are masked.
    """
    product_id = product_id or site.satellite.products.cmems_altimeter_l3_id
    if max_dist_km is None:
        max_dist_km = site.satellite.products.altimeter_max_dist_km
    if cache_dir is None:
        cache_dir = Path(site.satellite.cache_root) / "waves"
    key = cache_key(f"altimeter_{product_id}", start, end)
    lat, lon = site_lat_lon(site)

    def _fetch() -> pd.DataFrame:
        cm = _require_copernicusmarine()
        lonmin, latmin, lonmax, latmax = bbox_from_site(site)
        # Widen the bbox so sparse tracks intersect the window; the haversine cut
        # below enforces the precise distance limit.
        pad = max(0.5, max_dist_km / 111.0)
        nc_path = Path(cache_dir) / key.replace(".parquet", ".nc")
        _subset_nc(
            cm,
            dataset_id=product_id,
            variables=[ALT_SWH_CANDIDATES[0]],
            bbox=(lonmin - pad, latmin - pad, lonmax + pad, latmax + pad),
            start=start,
            end=end,
            nc_path=nc_path,
        )
        return _altimeter_df_from_nc(nc_path, lat, lon, max_dist_km, product_id)

    return load_from_cache_or(cache_dir, key, _fetch)


def _altimeter_df_from_nc(
    nc_path: Path, lat: float, lon: float, max_dist_km: float, product_id: str
) -> pd.DataFrame:
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        df = ds.to_dataframe().reset_index()
    if df.empty:
        return _empty_altimeter()
    lat_c = next((c for c in ("latitude", "lat", "nav_lat") if c in df.columns), None)
    lon_c = next((c for c in ("longitude", "lon", "nav_lon") if c in df.columns), None)
    swh_c = next((c for c in ALT_SWH_CANDIDATES if c in df.columns), None)
    if not (lat_c and lon_c and swh_c) or "time" not in df.columns:
        log.warning("altimeter nc missing expected columns; got %s", list(df.columns))
        return _empty_altimeter()

    df = df.dropna(subset=[swh_c, lat_c, lon_c])
    if df.empty:
        return _empty_altimeter()
    df["alt_dist_km"] = haversine_km(lat, lon, df[lat_c].to_numpy(), df[lon_c].to_numpy())
    df = df[df["alt_dist_km"] <= max_dist_km]
    if df.empty:
        return _empty_altimeter()

    mission_c = next((c for c in ("mission", "platform", "source") if c in df.columns), None)
    out = pd.DataFrame(
        {
            "alt_swh": df[swh_c].to_numpy(),
            "alt_dist_km": df["alt_dist_km"].to_numpy(),
            "alt_mission": df[mission_c].astype(str).to_numpy() if mission_c else product_id,
        },
        index=pd.to_datetime(df["time"], utc=True),
    )
    out.index.name = "time"
    return coerce_utc_index(out)


def latest_model_hs(site: Site, hours: int = 48) -> pd.DataFrame:
    """Convenience: last N hours of model Hs/Tp/Dp for quick sanity checks."""
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)
    return fetch_cmems_wave_model(site, start, end)
