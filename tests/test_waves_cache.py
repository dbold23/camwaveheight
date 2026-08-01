"""Unit test for satellite.waves cache path with a mocked copernicusmarine.

We monkeypatch `copernicusmarine.subset` to write a tiny NetCDF, then assert the
fetch parses it to the right columns/index and that the parquet cache is reused
(subset not called twice). No network or credentials.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from camwaveheight.satellite import waves
from camwaveheight.site import AOI, GeoLocation, SatelliteConfig, Site


def make_site(tmp_path) -> Site:
    return Site(
        name="t",
        buoy_id="201",
        location=GeoLocation(lat=32.867, lon=-117.257),
        aoi=AOI(bbox=(-117.30, 32.82, -117.21, 32.91)),
        satellite=SatelliteConfig(enabled=True, cache_root=str(tmp_path)),
    )


def _write_tiny_wave_nc(path: Path) -> None:
    import xarray as xr

    times = pd.date_range("2026-05-07", periods=4, freq="3h")
    lats = np.array([32.8, 32.9])
    lons = np.array([-117.30, -117.20])
    shape = (len(times), len(lats), len(lons))

    def fld(base):
        return (("time", "latitude", "longitude"), np.full(shape, base))

    ds = xr.Dataset(
        {"VHM0": fld(1.0), "VTPK": fld(12.0), "VMDR": fld(270.0)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    ds.to_netcdf(path)


def _install_fake_cm(monkeypatch, subset_fn):
    fake = types.ModuleType("copernicusmarine")
    fake.subset = subset_fn
    monkeypatch.setitem(sys.modules, "copernicusmarine", fake)
    return fake


def test_fetch_cmems_wave_model_parses_and_caches(tmp_path, monkeypatch):
    site = make_site(tmp_path)
    calls = {"n": 0}

    def subset(**kwargs):
        calls["n"] += 1
        out = Path(kwargs["output_directory"]) / kwargs["output_filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_tiny_wave_nc(out)

    _install_fake_cm(monkeypatch, subset)

    df = waves.fetch_cmems_wave_model(site, "2026-05-07", "2026-05-07T12:00:00")
    assert list(df.columns) == ["model_hs", "model_tp", "model_dp"]
    assert len(df) == 4
    assert df["model_hs"].iloc[0] == pytest.approx(1.0)
    assert df["model_tp"].iloc[0] == pytest.approx(12.0)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert calls["n"] == 1

    # Parquet cache written under <cache_root>/waves/
    cache_files = list((tmp_path / "waves").glob("cmemswave_*.parquet"))
    assert cache_files, "expected a parquet cache file"

    # Second call must hit the cache: break subset to prove it isn't called.
    def boom(**kwargs):
        raise RuntimeError("subset should not be called when cache exists")

    _install_fake_cm(monkeypatch, boom)
    df2 = waves.fetch_cmems_wave_model(site, "2026-05-07", "2026-05-07T12:00:00")
    pd.testing.assert_frame_equal(df, df2)


def test_require_copernicusmarine_missing(monkeypatch):
    # Simulate the package being absent.
    monkeypatch.setitem(sys.modules, "copernicusmarine", None)
    with pytest.raises(ModuleNotFoundError) as exc:
        waves._require_copernicusmarine()
    assert "sat-waves" in str(exc.value)


def test_empty_helpers_have_named_time_index():
    assert waves._empty_model().index.name == "time"
    assert list(waves._empty_altimeter().columns) == ["alt_swh", "alt_dist_km", "alt_mission"]
