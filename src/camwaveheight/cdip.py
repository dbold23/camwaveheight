"""CDIP buoy data fetcher via ERDDAP.

CDIP publishes wave parameters from all stations in a single aggregated
tabledap dataset, `wave_agg`, filtered by `station_id` (e.g. "201" for
Scripps Nearshore). We pull `waveHs`, `waveTp`, `waveTa` (zero-up-crossing
period), and `waveDp` at native ~30-min cadence.

Usage:
    from camwaveheight.cdip import fetch_cdip_params
    df = fetch_cdip_params("201", "2026-05-01", "2026-05-25")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import io

import pandas as pd
import requests
from erddapy import ERDDAP

log = logging.getLogger(__name__)

CDIP_ERDDAP = "https://erddap.cdip.ucsd.edu/erddap"
WAVE_DATASET = "wave_agg"
DEFAULT_VARS = ("waveHs", "waveTp", "waveTa", "waveDp")


def _iso(t: str | datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_cdip_params(
    station_id: str,
    start: str | datetime,
    end: str | datetime,
    variables: tuple[str, ...] = DEFAULT_VARS,
    cache_dir: str | Path | None = None,
    qc_primary_only: bool = True,
) -> pd.DataFrame:
    """Fetch CDIP parameter time series for a station between [start, end] UTC.

    Args:
        station_id: 3-digit CDIP station number, e.g. "201". Zero-pad if needed.
        start, end: ISO date strings or datetimes; interpreted as UTC.
        variables: ERDDAP variable names to pull. Defaults cover Hs, Tp, Tz, Dp.
        cache_dir: if set, parquet cache keyed by station + window.
        qc_primary_only: if True, keeps only `waveFlagPrimary=1` (good QC).

    Returns:
        DataFrame indexed by UTC timestamp with one column per requested variable.
    """
    stn = station_id.zfill(3)
    start_iso, end_iso = _iso(start), _iso(end)

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = f"cdip{stn}_{start_iso}_{end_iso}.parquet".replace(":", "")
        cache_path = cache_dir / cache_key
        if cache_path.exists():
            log.info("loading cached %s", cache_path)
            return pd.read_parquet(cache_path)

    e = ERDDAP(server=CDIP_ERDDAP, protocol="tabledap", response="csv")
    e.dataset_id = WAVE_DATASET
    e.variables = ["time", "station_id", *variables]
    constraints: dict[str, object] = {
        "time>=": start_iso,
        "time<=": end_iso,
        "station_id=": stn,
    }
    if qc_primary_only:
        constraints["waveFlagPrimary="] = 1
    e.constraints = constraints

    url = e.get_download_url(response="csv")
    log.info("fetching %s station=%s %s..%s", WAVE_DATASET, stn, start_iso, end_iso)
    log.debug("url=%s", url)
    resp = requests.get(url, timeout=60)
    if resp.status_code == 404 and "Your query produced no matching results" in resp.text:
        log.warning("no rows for %s in [%s, %s]", stn, start_iso, end_iso)
        return pd.DataFrame(columns=list(variables)).rename_axis("time")
    resp.raise_for_status()
    # ERDDAP CSV has a units row after the header — skip it.
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    if "station_id" in df.columns:
        df = df.drop(columns=["station_id"])

    if cache_dir is not None:
        df.to_parquet(cache_path)
        log.info("cached %s", cache_path)
    return df


def latest_hs(station_id: str, hours: int = 24) -> pd.DataFrame:
    """Convenience: last N hours of Hs/Tp for quick sanity checks."""
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(hours=hours)
    return fetch_cdip_params(station_id, start, end)
