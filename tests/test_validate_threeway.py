"""Unit tests for the three-way comparison in validate.py.

Synthetic series with known linear relationships → closed-form metrics.
All offline; no network or credentials. Also pins that the shared `_metrics`
refactor leaves the two-way `fit_train_test` numerics unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from camwaveheight import validate


def _frame(vals, *, start="2026-05-07T00:00:00Z", freq="30min", col="v"):
    idx = pd.date_range(start, periods=len(vals), freq=freq, tz="UTC")
    return pd.DataFrame({col: list(vals)}, index=idx)


def test_metrics_perfect():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    m = validate._metrics(ref, ref)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["bias"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["n"] == 4


def test_metrics_constant_offset():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    m = validate._metrics(ref + 0.5, ref)
    assert m["bias"] == pytest.approx(0.5)
    assert m["rmse"] == pytest.approx(0.5)


def test_metrics_drops_nonfinite():
    pred = np.array([1.0, np.nan, 3.0])
    ref = np.array([1.0, 2.0, 3.0])
    m = validate._metrics(pred, ref)
    assert m["n"] == 2
    assert m["rmse"] == pytest.approx(0.0)


def test_metrics_empty():
    m = validate._metrics(np.array([]), np.array([]))
    assert m["n"] == 0
    assert np.isnan(m["rmse"])


def test_merge_source_asof():
    paired = _frame([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], col="waveHs")
    paired["hs_pred_m"] = paired["waveHs"] + 0.1
    model = _frame([1.0, 1.5, 2.0], freq="90min", col="model_hs")
    merged = validate.merge_source(paired, model, "model_hs", tolerance="90min")
    assert "model_hs" in merged.columns
    assert merged["model_hs"].notna().sum() >= 3


def test_merge_source_missing_column_raises():
    paired = _frame([1.0, 1.1], col="waveHs")
    with pytest.raises(KeyError):
        validate.merge_source(paired, _frame([1.0], col="x"), "model_hs")


def test_compare_sources_known_metrics():
    paired = _frame([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], col="waveHs")
    paired["hs_pred_m"] = paired["waveHs"] + 0.1  # +0.1 m constant bias
    paired["model_hs"] = paired["waveHs"]  # perfect
    table = validate.compare_sources(
        paired, sources=("hs_pred_m", "model_hs", "alt_swh"), reference="waveHs"
    )
    assert list(table.index) == ["hs_pred_m", "model_hs"]  # alt_swh absent → skipped
    assert table.loc["hs_pred_m", "bias"] == pytest.approx(0.1)
    assert table.loc["hs_pred_m", "rmse"] == pytest.approx(0.1)
    assert table.loc["model_hs", "rmse"] == pytest.approx(0.0)
    assert table.loc["model_hs", "r2"] == pytest.approx(1.0)


def test_compare_sources_bad_reference_raises():
    with pytest.raises(KeyError):
        validate.compare_sources(_frame([1.0], col="waveHs"), reference="nope")


def test_align_three_way_end_to_end():
    cam = _frame(range(100, 100 + 12 * 10, 10), freq="15min", col="hs_px_4std")
    buoy = _frame([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], col="waveHs")
    buoy["waveTp"] = 10.0
    model = _frame([1.0, 1.3, 1.6], freq="90min", col="model_hs")
    out = validate.align_three_way(cam, buoy, model, source_col="model_hs")
    assert {"waveHs", "hs_px", "model_hs"}.issubset(out.columns)
    assert len(out) >= 1
    assert out["model_hs"].notna().any()


def test_fit_train_test_unchanged_by_refactor():
    # A clean linear cam→buoy relationship; the refactor must reproduce exact fit.
    n = 20
    buoy = _frame(np.linspace(0.8, 2.0, n), col="waveHs")
    buoy["waveTp"] = 11.0
    # hs_px such that waveHs = 2.0 * hs_px + 0.1  → recoverable coefficients
    cam = _frame((buoy["waveHs"].to_numpy() - 0.1) / 2.0, freq="30min", col="hs_px_4std")
    paired = validate.align_to_buoy(cam, buoy, cam_col="hs_px_4std")
    fit, out = validate.fit_train_test(paired, train_frac=0.7)
    assert fit.scale_m_per_px == pytest.approx(2.0, rel=1e-6)
    assert fit.bias_m == pytest.approx(0.1, abs=1e-6)
    assert fit.rmse_test_m == pytest.approx(0.0, abs=1e-9)
    assert fit.r2_test == pytest.approx(1.0, abs=1e-9)
