"""Environmental context panels: SST, wind, ocean color.

Small UTC-indexed parquets under `data/satellite/env/`, joined into a daily
`env_panel.parquet`. The default SST path is the no-auth **NOAA Coral Reef Watch
5 km daily** product via CoastWatch ERDDAP — it reuses the same `erddapy` flow as
`cdip.py`, so P2 needs no credentials. MUR (PODAAC, ~1 km) is available as a
higher-resolution alternative but needs an Earthdata `~/.netrc`.

Wind (ASCAT) and ocean color (Sentinel-3 OLCI / NASA OC) require provider creds
and are imported lazily; they raise an actionable error if their deps/creds are
absent.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

from camwaveheight.satellite.common import (
    cache_key,
    coerce_utc_index,
    iso_utc,
    load_from_cache_or,
    site_lat_lon,
)
from camwaveheight.site import Site

log = logging.getLogger(__name__)

# NOAA CoastWatch ERDDAP — Coral Reef Watch daily global 5 km SST (no auth).
CRW_ERDDAP = "https://coastwatch.pfeg.noaa.gov/erddap"
CRW_DATASET = "NOAA_DHW"
CRW_SST_VAR = "CRW_SST"


def fetch_sst(
    site: Site,
    start,
    end,
    source: str | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Daily SST nearest the site → column `sst_c` (°C), UTC-indexed.

    `source` defaults to `site.satellite.products.sst_dataset`. "CRW" uses the
    no-auth NOAA CoastWatch ERDDAP path; "MUR" uses the PODAAC ~1 km product
    (Earthdata login required).
    """
    source = (source or site.satellite.products.sst_dataset).upper()
    if cache_dir is None:
        cache_dir = Path(site.satellite.cache_root) / "env"
    key = cache_key(f"sst_{source}", start, end)

    if source == "CRW":
        return load_from_cache_or(cache_dir, key, lambda: _fetch_sst_crw(site, start, end))
    if source == "MUR":
        return load_from_cache_or(cache_dir, key, lambda: _fetch_sst_mur(site, start, end))
    raise ValueError(f"unknown SST source '{source}' (expected 'CRW' or 'MUR')")


def _fetch_sst_crw(site: Site, start, end) -> pd.DataFrame:
    """NOAA CRW daily 5 km SST via CoastWatch ERDDAP griddap (no auth)."""
    import requests
    from erddapy import ERDDAP

    lat, lon = site_lat_lon(site)
    e = ERDDAP(server=CRW_ERDDAP, protocol="griddap", response="csv")
    e.dataset_id = CRW_DATASET
    e.variables = [CRW_SST_VAR]
    e.constraints = {
        "time>=": iso_utc(start),
        "time<=": iso_utc(end),
        "latitude>=": lat,
        "latitude<=": lat,
        "longitude>=": lon,
        "longitude<=": lon,
    }
    url = e.get_download_url(response="csv")
    log.info("fetching CRW SST nearest (%.3f, %.3f)", lat, lon)
    resp = requests.get(url, timeout=120)
    if resp.status_code == 404 and "no matching results" in resp.text.lower():
        return _empty_sst()
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), skiprows=[1])  # drop ERDDAP units row
    if df.empty or CRW_SST_VAR not in df.columns:
        return _empty_sst()
    out = pd.DataFrame({"sst_c": pd.to_numeric(df[CRW_SST_VAR], errors="coerce").to_numpy()},
                       index=pd.to_datetime(df["time"], utc=True))
    out.index.name = "time"
    return coerce_utc_index(out).dropna()


def _fetch_sst_mur(site: Site, start, end) -> pd.DataFrame:
    """MUR SST (PODAAC ~1 km). Requires an Earthdata login (~/.netrc)."""
    from camwaveheight.satellite.creds import earthdata_netrc

    earthdata_netrc()  # raises with guidance if missing
    raise NotImplementedError(
        "MUR SST fetch is not wired yet — use the no-auth CRW path "
        "(products.sst_dataset='CRW') for P2. MUR access goes through PODAAC "
        "Harmony/OPeNDAP and is a P2+ extension."
    )


def fetch_wind(
    site: Site, start, end, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Near-surface wind nearest the site → `wind_speed` (m/s), `wind_dir` (deg).

    ASCAT scatterometer via PODAAC (Earthdata login). Lazily gated.
    """
    from camwaveheight.satellite.creds import earthdata_netrc

    earthdata_netrc()
    raise NotImplementedError(
        "ASCAT wind fetch is a P2+ extension (PODAAC Harmony). For wind context "
        "now, prefer CMEMS/ERA5; SST via CRW is the verified P2 path."
    )


def fetch_ocean_color(
    site: Site, start, end, cache_dir: str | Path | None = None
) -> pd.DataFrame:
    """Chlorophyll / turbidity nearest the site → `chl_mg_m3`, `turbidity`.

    Sentinel-3 OLCI (300 m) via STAC, or NASA OC L3 (4 km). Lazily gated on the
    `sat-env` extra (`pystac-client` / `odc-stac`).
    """
    try:
        import pystac_client  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'Ocean-color needs the sat-env extra:  pip install -e ".[sat-env]"'
        ) from exc
    raise NotImplementedError(
        "Ocean-color fetch is a P2+ extension (STAC OLCI). SST via CRW is the "
        "verified P2 path."
    )


def build_env_panel(
    site: Site,
    start,
    end,
    variables: tuple[str, ...] = ("sst",),
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Outer-join requested env variables onto a daily UTC index → `env_panel`.

    `variables` ⊆ {"sst", "wind", "chl"}. Each is fetched independently and an
    empty/failed source contributes no columns (logged), so a partial panel still
    builds. Result is cached as `env_panel.parquet`.
    """
    if cache_dir is None:
        cache_dir = Path(site.satellite.cache_root) / "env"
    fetchers = {"sst": fetch_sst, "wind": fetch_wind, "chl": fetch_ocean_color}
    frames = []
    for var in variables:
        fn = fetchers.get(var)
        if fn is None:
            log.warning("build_env_panel: unknown variable '%s' skipped", var)
            continue
        try:
            df = fn(site, start, end, cache_dir=cache_dir)
        except NotImplementedError as exc:
            log.warning("env '%s' unavailable: %s", var, exc)
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        panel = pd.DataFrame()
    else:
        panel = frames[0]
        for df in frames[1:]:
            panel = panel.join(df, how="outer")
        # Collapse to a daily grid.
        panel = panel.resample("1D").mean()

    out_path = Path(cache_dir) / "env_panel.parquet"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if not panel.empty:
        panel.to_parquet(out_path)
        log.info("wrote %s (%d days, cols=%s)", out_path, len(panel), list(panel.columns))
    return panel


def _empty_sst() -> pd.DataFrame:
    return pd.DataFrame(columns=["sst_c"]).rename_axis("time")
