"""Shared helpers for the satellite subpackage.

Mirrors the `cdip.py` fetch/cache conventions so every satellite fetcher reads
the same way:
  - UTC coercion of timestamps (`iso_utc`, `coerce_utc_index`)
  - deterministic parquet cache keys (`cache_key`) — same `replace(":", "")`
    scheme as `cdip.fetch_cdip_params`
  - "read cache if exists else fetch and persist" (`load_from_cache_or`)
  - AOI/bbox derivation from the site config (`bbox_from_site`)
  - great-circle distance for along-track altimeter filtering (`haversine_km`)

Only light deps (numpy/pandas) are imported here; provider SDKs are imported
lazily by the callers in waves/shoreline/environment.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from camwaveheight.site import Site

log = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0088


def iso_utc(t: str | datetime | pd.Timestamp) -> str:
    """Coerce any timestamp-like to a UTC `YYYY-mm-ddTHH:MM:SSZ` string.

    Identical semantics to `cdip._iso` so cache keys line up across modules.
    """
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def cache_key(prefix: str, start, end, suffix: str = ".parquet") -> str:
    """Deterministic cache filename: `<prefix>_<startISO>_<endISO><suffix>`.

    The colons are stripped exactly as in `cdip.fetch_cdip_params` so filenames
    are filesystem-safe and reproducible.
    """
    return f"{prefix}_{iso_utc(start)}_{iso_utc(end)}{suffix}".replace(":", "")


def bbox_from_site(site: Site, default_radius_deg: float = 0.25) -> tuple[float, float, float, float]:
    """Return (lonmin, latmin, lonmax, latmax) for subsetting.

    Prefers `site.aoi.bbox`; otherwise builds a square of ±radius degrees around
    `site.location`. Raises if neither is available.
    """
    if site.aoi is not None and site.aoi.bbox is not None:
        lonmin, latmin, lonmax, latmax = (float(v) for v in site.aoi.bbox)
        return lonmin, latmin, lonmax, latmax
    if site.location is None:
        raise ValueError(
            f"site '{site.name}' has neither aoi.bbox nor location; cannot derive a bbox"
        )
    r = site.aoi.subset_radius_deg if site.aoi is not None else default_radius_deg
    lat, lon = site.location.lat, site.location.lon
    return (lon - r, lat - r, lon + r, lat + r)


def site_lat_lon(site: Site) -> tuple[float, float]:
    """Point coordinates for nearest-grid-cell selection / distance filtering.

    Uses `site.location` when present, else the AOI bbox centroid.
    """
    if site.location is not None:
        return site.location.lat, site.location.lon
    lonmin, latmin, lonmax, latmax = bbox_from_site(site)
    return (latmin + latmax) / 2.0, (lonmin + lonmax) / 2.0


def coerce_utc_index(df: pd.DataFrame, time_col: str | None = None) -> pd.DataFrame:
    """Return `df` with a sorted UTC `DatetimeIndex` at ns precision.

    Matches the precision normalization in `validate.align_to_buoy` so satellite
    frames merge_asof cleanly against cam/buoy timelines.
    """
    df = df.copy()
    if time_col is not None and time_col in df.columns:
        df = df.set_index(time_col)
    df.index = pd.to_datetime(df.index, utc=True).as_unit("ns")
    df.index.name = df.index.name or "time"
    return df.sort_index()


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays for the 2nd point."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def load_from_cache_or(
    cache_dir: str | Path | None,
    key: str,
    fetch: Callable[[], pd.DataFrame],
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Read `cache_dir/key` parquet if present, else `fetch()`, persist, return.

    Mirrors the cache idiom in `cdip.fetch_cdip_params`. Empty results are
    returned but never cached (so a transient empty window can be retried).
    """
    if cache_dir is None:
        return fetch()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / key
    if cache_path.exists():
        log.info("loading cached %s", cache_path)
        return pd.read_parquet(cache_path)
    df = fetch()
    if save and df is not None and not df.empty:
        df.to_parquet(cache_path)
        log.info("cached %s", cache_path)
    return df
