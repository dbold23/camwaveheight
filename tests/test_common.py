"""Unit tests for satellite.common — cache keys, bbox, haversine, UTC coercion.

All offline; no network or credentials.
"""

from __future__ import annotations

import pandas as pd
import pytest

from camwaveheight.satellite import common
from camwaveheight.site import AOI, GeoLocation, SatelliteConfig, Site


def make_site(**kw) -> Site:
    base = dict(
        name="t",
        buoy_id="201",
        location=GeoLocation(lat=32.867, lon=-117.257),
        aoi=AOI(bbox=(-117.30, 32.82, -117.21, 32.91)),
        satellite=SatelliteConfig(enabled=True),
    )
    base.update(kw)
    return Site(**base)


def test_iso_utc_naive_is_treated_as_utc():
    assert common.iso_utc("2026-05-07") == "2026-05-07T00:00:00Z"


def test_iso_utc_aware_is_converted():
    out = common.iso_utc(pd.Timestamp("2026-05-07T12:00:00", tz="US/Pacific"))
    assert out == "2026-05-07T19:00:00Z"


def test_cache_key_strips_colons_like_cdip():
    k = common.cache_key("cmemswave_x", "2026-05-07T03:00:00", "2026-05-08T00:00:00")
    assert ":" not in k
    assert k.startswith("cmemswave_x_") and k.endswith(".parquet")


def test_cache_key_custom_suffix():
    assert common.cache_key("alt", "2026-05-07", "2026-05-08", suffix=".nc").endswith(".nc")


def test_bbox_from_aoi_bbox():
    assert common.bbox_from_site(make_site()) == (-117.30, 32.82, -117.21, 32.91)


def test_bbox_from_location_radius():
    s = make_site(aoi=AOI(bbox=None, subset_radius_deg=0.1))
    lonmin, latmin, lonmax, latmax = common.bbox_from_site(s)
    assert lonmin == pytest.approx(-117.357)
    assert latmin == pytest.approx(32.767)
    assert lonmax == pytest.approx(-117.157)
    assert latmax == pytest.approx(32.967)


def test_bbox_requires_location_or_bbox():
    s = make_site(aoi=None, location=None)
    with pytest.raises(ValueError):
        common.bbox_from_site(s)


def test_site_lat_lon_prefers_location():
    assert common.site_lat_lon(make_site()) == (32.867, -117.257)


def test_site_lat_lon_falls_back_to_bbox_centroid():
    s = make_site(location=None)
    lat, lon = common.site_lat_lon(s)
    assert lat == pytest.approx((32.82 + 32.91) / 2)
    assert lon == pytest.approx((-117.30 + -117.21) / 2)


def test_haversine_one_degree_latitude():
    assert common.haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.5)


def test_haversine_zero_distance():
    assert common.haversine_km(32.0, -117.0, 32.0, -117.0) == pytest.approx(0.0, abs=1e-6)


def test_coerce_utc_index_sorts_and_normalizes():
    df = pd.DataFrame({"x": [1, 2]}, index=pd.to_datetime(["2026-05-08", "2026-05-07"]))
    out = common.coerce_utc_index(df)
    assert out.index.is_monotonic_increasing
    assert str(out.index.tz) == "UTC"
    assert out.index.dtype == "datetime64[ns, UTC]"


def test_load_from_cache_or_roundtrip(tmp_path):
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        return pd.DataFrame({"a": [1.0]}, index=pd.to_datetime(["2026-05-07"], utc=True))

    a = common.load_from_cache_or(tmp_path, "x.parquet", fetch)
    b = common.load_from_cache_or(tmp_path, "x.parquet", fetch)  # served from cache
    assert calls["n"] == 1
    assert (tmp_path / "x.parquet").exists()
    pd.testing.assert_frame_equal(a, b)


def test_load_from_cache_or_does_not_cache_empty(tmp_path):
    common.load_from_cache_or(tmp_path, "e.parquet", lambda: pd.DataFrame())
    assert not (tmp_path / "e.parquet").exists()


def test_load_from_cache_or_no_cache_dir_passthrough():
    df = common.load_from_cache_or(None, "k.parquet", lambda: pd.DataFrame({"a": [1]}))
    assert list(df["a"]) == [1]
