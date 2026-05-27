"""Regress pixel-space Hs against CDIP buoy Hs, evaluate on held-out window.

Workflow:
  1. Align cam Hs(t) to CDIP Hs(t) (both UTC, asof merge with tolerance).
  2. Split into train / held-out windows by timestamp.
  3. Fit: meter_hs = a * pixel_hs + b on the train window.
  4. Apply to held-out, compute RMSE / bias / R^2 / scatter index.
  5. Persist coefficients to the site config.

Plot outputs go to `reports/`:
  - validation_v1_timeseries.png  (cam vs buoy over time)
  - validation_v1_scatter.png     (1:1 with fit line, train + held-out)
  - validation_v1_residuals.png   (residuals vs time and vs swell period)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FitResult:
    scale_m_per_px: float
    bias_m: float
    n_train: int
    n_test: int
    rmse_train_m: float
    rmse_test_m: float
    bias_test_m: float
    r2_test: float
    scatter_index_test: float


def align_to_buoy(
    cam_hs: pd.DataFrame,
    buoy: pd.DataFrame,
    tolerance: str = "15min",
    cam_col: str = "hs_px_4std",
) -> pd.DataFrame:
    """asof-merge cam Hs into buoy timestamps (buoy is at 30-min cadence)."""
    cam = cam_hs[[cam_col]].rename(columns={cam_col: "hs_px"}).sort_index()
    buoy = buoy[["waveHs", "waveTp"]].sort_index()
    # Normalize datetime precision — CDIP comes back as us, cam stats are ns.
    cam.index = pd.to_datetime(cam.index, utc=True).as_unit("ns")
    buoy.index = pd.to_datetime(buoy.index, utc=True).as_unit("ns")
    out = pd.merge_asof(
        buoy,
        cam,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(tolerance),
        direction="nearest",
    )
    return out.dropna(subset=["hs_px", "waveHs"])


def fit_train_test(
    paired: pd.DataFrame,
    train_frac: float = 0.7,
) -> tuple[FitResult, pd.DataFrame]:
    """Time-ordered train/test split, linear regression on train, evaluate on test.

    Returns the fit object and the input DataFrame with `hs_pred_m` filled in
    for all rows (train + test) using the train-fit coefficients.
    """
    if len(paired) < 6:
        raise ValueError(f"need at least 6 paired samples, got {len(paired)}")
    n = len(paired)
    n_train = int(n * train_frac)
    train = paired.iloc[:n_train]
    test = paired.iloc[n_train:]

    x = train["hs_px"].to_numpy()
    y = train["waveHs"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    (scale, bias), *_ = np.linalg.lstsq(A, y, rcond=None)

    paired = paired.copy()
    paired["hs_pred_m"] = scale * paired["hs_px"] + bias

    resid_train = paired.iloc[:n_train]["hs_pred_m"] - train["waveHs"]
    resid_test = paired.iloc[n_train:]["hs_pred_m"] - test["waveHs"]

    rmse_train = float(np.sqrt((resid_train**2).mean()))
    rmse_test = float(np.sqrt((resid_test**2).mean()))
    bias_test = float(resid_test.mean())
    ss_res = float((resid_test**2).sum())
    ss_tot = float(((test["waveHs"] - test["waveHs"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    si = rmse_test / float(test["waveHs"].mean()) if test["waveHs"].mean() > 0 else float("nan")

    fit = FitResult(
        scale_m_per_px=float(scale),
        bias_m=float(bias),
        n_train=len(train),
        n_test=len(test),
        rmse_train_m=rmse_train,
        rmse_test_m=rmse_test,
        bias_test_m=bias_test,
        r2_test=r2,
        scatter_index_test=si,
    )
    log.info("fit: %s", fit)
    return fit, paired


def plot_validation(
    paired: pd.DataFrame,
    fit: FitResult,
    out_dir: str | Path = "reports",
    tag: str = "v1",
) -> dict[str, Path]:
    """Three diagnostic plots: time series, scatter, residuals."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = fit.n_train

    # Timeseries
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(paired.index, paired["waveHs"], "o-", color="C0", label="CDIP 201 Hs", markersize=3)
    ax.plot(paired.index, paired["hs_pred_m"], ".-", color="C3", label="cam Hs (fit)", markersize=3)
    ax.axvline(paired.index[n_train - 1], color="k", ls="--", alpha=0.4, label="train | test")
    ax.set_ylabel("Hs (m)")
    ax.set_title(f"CamWaveHeight — validation {tag}: RMSE_test={fit.rmse_test_m:.3f} m, R²={fit.r2_test:.2f}")
    ax.legend()
    ts_path = out_dir / f"validation_{tag}_timeseries.png"
    fig.tight_layout()
    fig.savefig(ts_path, dpi=120)
    plt.close(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    train = paired.iloc[:n_train]
    test = paired.iloc[n_train:]
    ax.scatter(train["waveHs"], train["hs_pred_m"], s=18, alpha=0.6, label=f"train (n={fit.n_train})")
    ax.scatter(test["waveHs"], test["hs_pred_m"], s=22, alpha=0.8, color="C3", label=f"test (n={fit.n_test})")
    lim = [
        float(min(paired["waveHs"].min(), paired["hs_pred_m"].min())) * 0.9,
        float(max(paired["waveHs"].max(), paired["hs_pred_m"].max())) * 1.1,
    ]
    ax.plot(lim, lim, "k--", alpha=0.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("CDIP 201 Hs (m)")
    ax.set_ylabel("cam-derived Hs (m)")
    ax.set_title(f"{tag}: RMSE_test={fit.rmse_test_m:.3f} m")
    ax.legend()
    sc_path = out_dir / f"validation_{tag}_scatter.png"
    fig.tight_layout()
    fig.savefig(sc_path, dpi=120)
    plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(11, 3))
    resid = paired["hs_pred_m"] - paired["waveHs"]
    ax.plot(paired.index, resid, ".", color="C2", markersize=4)
    ax.axhline(0, color="k", ls=":", alpha=0.4)
    ax.axhline(0.25, color="r", ls=":", alpha=0.4, label="±25 cm target")
    ax.axhline(-0.25, color="r", ls=":", alpha=0.4)
    ax.set_ylabel("residual (m)")
    ax.set_title("cam − buoy residuals over time")
    ax.legend()
    res_path = out_dir / f"validation_{tag}_residuals.png"
    fig.tight_layout()
    fig.savefig(res_path, dpi=120)
    plt.close(fig)

    return {"timeseries": ts_path, "scatter": sc_path, "residuals": res_path}


def save_summary(fit: FitResult, train_window: tuple[str, str], out_path: str | Path) -> None:
    """Write a small JSON next to the plots for quick inspection."""
    import json

    payload = {**asdict(fit), "fit_train_window": list(train_window)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2))
