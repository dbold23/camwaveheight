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


def _metrics(pred, ref) -> dict[str, float]:
    """RMSE / bias / R^2 / scatter-index of `pred` vs reference `ref`.

    Non-finite pairs are dropped. Shared by `fit_train_test` and `compare_sources`
    so every source (cam, model, altimeter) is scored with identical math.
    """
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(ref)
    pred, ref = pred[mask], ref[mask]
    n = int(pred.size)
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "bias": float("nan"),
                "r2": float("nan"), "scatter_index": float("nan")}
    resid = pred - ref
    rmse = float(np.sqrt(np.mean(resid**2)))
    bias = float(np.mean(resid))
    ss_res = float(np.sum(resid**2))
    ref_mean = float(np.mean(ref))
    ss_tot = float(np.sum((ref - ref_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    si = rmse / ref_mean if ref_mean > 0 else float("nan")
    return {"n": n, "rmse": rmse, "bias": bias, "r2": r2, "scatter_index": si}


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

    # Same metric math as compare_sources, via the shared _metrics helper.
    train_m = _metrics(paired.iloc[:n_train]["hs_pred_m"], train["waveHs"])
    test_m = _metrics(paired.iloc[n_train:]["hs_pred_m"], test["waveHs"])
    rmse_train = train_m["rmse"]
    rmse_test = test_m["rmse"]
    bias_test = test_m["bias"]
    r2 = test_m["r2"]
    si = test_m["scatter_index"]

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


# --------------------------------------------------------------------------- #
# Three-way comparison: cam Hs vs CDIP buoy vs a satellite/model source.
# Additive — the two-way cam-vs-buoy path above is untouched.
# --------------------------------------------------------------------------- #


def merge_source(
    paired: pd.DataFrame,
    source: pd.DataFrame,
    source_col: str,
    out_col: str | None = None,
    tolerance: str = "90min",
) -> pd.DataFrame:
    """asof-merge one extra source column onto an existing buoy-timeline frame.

    `paired` is the output of `align_to_buoy` / `fit_train_test` (UTC index on buoy
    times). `source` is any UTC-indexed satellite/model frame (e.g. the CMEMS model
    at 3-hourly cadence — hence the larger default tolerance). Rows with no match
    within `tolerance` get NaN in `out_col`.
    """
    out_col = out_col or source_col
    if source_col not in source.columns:
        raise KeyError(f"source has no column '{source_col}'; has {list(source.columns)}")
    src = source[[source_col]].rename(columns={source_col: out_col}).sort_index().copy()
    src.index = pd.to_datetime(src.index, utc=True).as_unit("ns")
    src = src[~src.index.duplicated(keep="first")].dropna()
    base = paired.copy()
    base.index = pd.to_datetime(base.index, utc=True).as_unit("ns")
    base = base.sort_index()
    return pd.merge_asof(
        base,
        src,
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta(tolerance),
        direction="nearest",
    )


def align_three_way(
    cam_hs: pd.DataFrame,
    buoy: pd.DataFrame,
    source: pd.DataFrame,
    source_col: str = "model_hs",
    cam_col: str = "hs_px_4std",
    cam_tolerance: str = "15min",
    source_tolerance: str = "90min",
) -> pd.DataFrame:
    """Buoy-timeline frame carrying cam Hs (px) and a model/sat source.

    Convenience: `align_to_buoy(cam_hs, buoy)` then `merge_source` for the
    satellite/model column. Returns waveHs, waveTp, hs_px, <source_col>. For a
    fitted comparison, fit cam→meters first with `fit_train_test`, then add more
    sources (e.g. altimeter) onto that result with `merge_source`.
    """
    paired = align_to_buoy(cam_hs, buoy, tolerance=cam_tolerance, cam_col=cam_col)
    return merge_source(paired, source, source_col, tolerance=source_tolerance)


def compare_sources(
    paired: pd.DataFrame,
    sources: tuple[str, ...] = ("hs_pred_m", "model_hs", "alt_swh"),
    reference: str = "waveHs",
) -> pd.DataFrame:
    """Per-source RMSE / bias / R² / scatter-index against `reference`.

    Returns a DataFrame indexed by source name with columns n, rmse, bias, r2,
    scatter_index. Sources absent from `paired` are skipped; each present source is
    scored over the rows where both it and the reference are finite (so a sparse
    altimeter is judged only on its few valid rows).
    """
    if reference not in paired.columns:
        raise KeyError(
            f"reference '{reference}' not in paired columns {list(paired.columns)}"
        )
    rows = []
    for src in sources:
        if src not in paired.columns:
            log.info("compare_sources: skipping absent source '%s'", src)
            continue
        rows.append({"source": src, **_metrics(paired[src], paired[reference])})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("source")
    return out


def plot_three_way(
    paired: pd.DataFrame,
    sources: tuple[str, ...] = ("hs_pred_m", "model_hs", "alt_swh"),
    reference: str = "waveHs",
    out_dir: str | Path = "reports",
    tag: str = "threeway",
) -> dict[str, Path]:
    """Time-series overlay + scatter of every available source vs the buoy.

    Styled like `plot_validation`. Returns the written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    present = [s for s in sources if s in paired.columns]
    style = {
        "hs_pred_m": ("C3", "cam Hs (fit)", ".-"),
        "model_hs": ("C1", "CMEMS model", ".-"),
        "alt_swh": ("C2", "altimeter SWH", "x"),
    }

    # Timeseries
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(paired.index, paired[reference], "o-", color="C0", ms=3, label=f"{reference} (CDIP)")
    for s in present:
        color, label, marker = style.get(s, (None, s, ".-"))
        ax.plot(paired.index, paired[s], marker, color=color, ms=5, alpha=0.85, label=label)
    ax.set_ylabel("Hs (m)")
    ax.set_xlabel("UTC")
    ax.set_title(f"Three-way Hs vs CDIP — {tag}")
    ax.legend()
    ts_path = out_dir / f"threeway_{tag}_timeseries.png"
    fig.tight_layout()
    fig.savefig(ts_path, dpi=120)
    plt.close(fig)

    # Scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    for s in present:
        color, label, _ = style.get(s, (None, s, ".-"))
        sub = paired[[reference, s]].dropna()
        ax.scatter(sub[reference], sub[s], s=18, alpha=0.7, color=color, label=label)
    allv = pd.concat([paired[reference], *[paired[s] for s in present]]).dropna()
    if not allv.empty:
        lim = [float(allv.min()) * 0.9, float(allv.max()) * 1.1]
        ax.plot(lim, lim, "k--", alpha=0.5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xlabel(f"{reference} (m)")
    ax.set_ylabel("source Hs (m)")
    ax.set_title(f"{tag}: sources vs CDIP")
    ax.legend()
    sc_path = out_dir / f"threeway_{tag}_scatter.png"
    fig.tight_layout()
    fig.savefig(sc_path, dpi=120)
    plt.close(fig)

    return {"timeseries": ts_path, "scatter": sc_path}
