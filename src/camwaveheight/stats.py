"""Wave statistics from a η(t) time series.

Two implementations:
  - 4*std method (cheap, robust to gaps): Hs = 4 * std(detrended η)
  - Zero-up-crossing method (CDIP-standard): Hs = mean of top 1/3 wave heights

We compute both per rolling window so the regression can pick the cleaner one.

All "Hs" returned by this module is in *pixel* space until `validate.py`
fits the regression to meters.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

log = logging.getLogger(__name__)


def _resample_uniform(
    eta: pd.Series, fs_hz: float, max_gap_sec: float = 10.0
) -> pd.Series:
    """Interpolate eta(t) onto a uniform fs_hz grid; only gaps ≤ max_gap_sec get filled.

    Critically, gaps larger than max_gap_sec (e.g. multi-hour night blackouts)
    remain NaN so downstream rolling-window coverage checks see them as missing.
    """
    eta = eta.dropna()
    if eta.empty:
        return eta
    t0, t1 = eta.index.min(), eta.index.max()
    dt = pd.Timedelta(seconds=1 / fs_hz)
    grid = pd.date_range(t0, t1, freq=dt)
    interp_limit = max(1, int(max_gap_sec * fs_hz))
    return (
        eta.reindex(eta.index.union(grid))
        .interpolate("time", limit=interp_limit, limit_direction="both")
        .reindex(grid)
    )


def detrend_lowpass(eta_px: pd.Series, fs_hz: float, lowpass_sec: float = 30.0) -> pd.Series:
    """Subtract a slow-moving baseline (tide, glare drift) via Butterworth lowpass.

    The lowpass cutoff at 1/lowpass_sec preserves all gravity-wave energy
    (periods 4–25 s) and removes everything slower than 30 s.
    """
    eta_u = _resample_uniform(eta_px, fs_hz)
    if eta_u.empty:
        return eta_u
    cutoff_hz = 1.0 / lowpass_sec
    nyq = fs_hz / 2
    sos = butter(N=4, Wn=cutoff_hz / nyq, btype="low", output="sos")
    baseline = sosfiltfilt(sos, eta_u.to_numpy())
    return pd.Series(eta_u.to_numpy() - baseline, index=eta_u.index, name="eta_detrend_px")


def hs_4std(detrended: pd.Series) -> float:
    """Hs = 4 * std(detrended η)."""
    arr = detrended.dropna().to_numpy()
    if arr.size < 100:
        return float("nan")
    return float(4.0 * np.std(arr))


def hs_zero_crossing(detrended: pd.Series) -> float:
    """Hs = mean of top 1/3 individual wave heights from zero-up-crossings."""
    arr = detrended.dropna().to_numpy()
    if arr.size < 100:
        return float("nan")
    # zero up-crossings: indices where signal crosses zero going positive
    zc = np.where((arr[:-1] < 0) & (arr[1:] >= 0))[0]
    if zc.size < 4:
        return float("nan")
    heights = []
    for i in range(zc.size - 1):
        seg = arr[zc[i] : zc[i + 1]]
        heights.append(seg.max() - seg.min())
    heights = np.array(heights)
    if heights.size < 3:
        return float("nan")
    top_third = np.sort(heights)[-max(1, len(heights) // 3) :]
    return float(top_third.mean())


def tp_welch(detrended: pd.Series, fs_hz: float) -> float:
    """Peak period via Welch PSD."""
    arr = detrended.dropna().to_numpy()
    if arr.size < 256:
        return float("nan")
    nperseg = min(arr.size, 512)
    f, pxx = welch(arr, fs=fs_hz, nperseg=nperseg)
    f, pxx = f[1:], pxx[1:]  # drop DC
    if pxx.size == 0:
        return float("nan")
    return float(1.0 / f[int(np.argmax(pxx))])


def rolling_hs(
    eta_px: pd.Series,
    fs_hz: float = 10.0,
    window_sec: int = 1020,
    step_sec: int = 300,
    lowpass_sec: float = 30.0,
) -> pd.DataFrame:
    """Rolling-window Hs/Tp in pixel space.

    Defaults: 17-min (CDIP-standard) windows, stepped every 5 min.
    """
    eta_u = _resample_uniform(eta_px, fs_hz).interpolate("time", limit=int(fs_hz * 5))

    cutoff_hz = 1.0 / lowpass_sec
    nyq = fs_hz / 2
    sos = butter(N=4, Wn=cutoff_hz / nyq, btype="low", output="sos")

    out_rows = []
    t0 = eta_u.index.min()
    t1 = eta_u.index.max() - pd.Timedelta(seconds=window_sec)
    if pd.isna(t0) or t0 >= t1:
        return pd.DataFrame()
    step = pd.Timedelta(seconds=step_sec)
    window = pd.Timedelta(seconds=window_sec)
    cur = t0
    while cur <= t1:
        win = eta_u.loc[cur : cur + window]
        # Need at least 70 % coverage for a valid stat
        if win.notna().mean() > 0.7 and win.size > 64:
            arr = win.to_numpy()
            arr = arr - sosfiltfilt(sos, arr)
            det = pd.Series(arr, index=win.index)
            hs4 = hs_4std(det)
            hszc = hs_zero_crossing(det)
            tp = tp_welch(det, fs_hz)
            out_rows.append(
                {
                    "window_start": cur,
                    "window_end": cur + window,
                    "hs_px_4std": hs4,
                    "hs_px_zc": hszc,
                    "tp_s": tp,
                    "n_samples": int(win.notna().sum()),
                }
            )
        cur += step
    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    df["window_mid"] = df["window_start"] + (df["window_end"] - df["window_start"]) / 2
    return df.set_index("window_mid")
