"""End-to-end orchestrator for one site.

run_pipeline:
  1. Load site config.
  2. Run wave_detect across all segments → η_px(t) (cached parquet).
  3. Compute rolling Hs_px (stats.py).
  4. Pull CDIP buoy Hs over the same span.
  5. Time-align, train/test split, regress, plot, persist coefficients.

Idempotent: η extraction caches; CDIP fetch caches; only the final fit + plots
are recomputed each run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from camwaveheight import cdip, stats, validate, wave_detect
from camwaveheight.site import Site

log = logging.getLogger(__name__)


def _save_cam_only(hs_df: pd.DataFrame, tag: str, out_dir: str | Path = "reports") -> Path:
    """Cam-only Hs_px / Tp diagnostic — runs before buoy data is available."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cam_only_{tag}.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(hs_df.index, hs_df["hs_px_4std"], ".-", label="Hs_px (4·std)")
    axes[0].plot(hs_df.index, hs_df["hs_px_zc"], ".-", color="C2", alpha=0.6, label="Hs_px (zero-cross)")
    axes[0].set_ylabel("Hs (pixels)")
    axes[0].legend()
    axes[0].set_title(f"cam-only diagnostic — {tag} (no CDIP overlap yet)")
    axes[1].plot(hs_df.index, hs_df["tp_s"], ".-", color="C1")
    axes[1].set_ylabel("Tp (s)")
    axes[1].set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_pipeline(
    site_path: str,
    out_root: str = "data/raw",
    cache_eta: str = "data/eta/eta_px.parquet",
    cdip_cache: str = "data/cdip",
    train_frac: float = 0.7,
    tag: str = "v1",
    fs_hz: float = 10.0,
    window_sec: int = 1020,
    step_sec: int = 300,
) -> dict:
    site = Site.load(site_path)
    if site.calibration.wave_roi is None:
        raise RuntimeError(
            f"{site.name}: wave_roi is not set. Run `cwh set-roi --site {site_path} ...` first."
        )
    roi = site.calibration.wave_roi
    log.info("running pipeline for %s; ROI=%s", site.name, roi)

    eta_df = wave_detect.extract_eta_for_site(
        site.name,
        roi,
        out_root=out_root,
        cache_path=cache_eta,
        sample_every_n_frames=3,  # ~10 Hz from 30 fps
    )
    if eta_df.empty:
        raise RuntimeError("no eta samples extracted — check recording and ROI")
    log.info("eta: %d samples, %s..%s", len(eta_df), eta_df.index.min(), eta_df.index.max())

    hs_df = stats.rolling_hs(
        eta_df["eta_px"],
        fs_hz=fs_hz,
        window_sec=window_sec,
        step_sec=step_sec,
    )
    if hs_df.empty:
        raise RuntimeError("rolling_hs produced no windows — need at least window_sec of footage")
    log.info("rolling Hs: %d windows", len(hs_df))

    # Widen the buoy query by ±2h so even short cam spans pair against multiple
    # CDIP samples (CDIP is 30-min cadence).
    t0 = hs_df.index.min().floor("h") - pd.Timedelta(hours=2)
    t1 = hs_df.index.max().ceil("h") + pd.Timedelta(hours=2)
    buoy = cdip.fetch_cdip_params(site.buoy_id, t0, t1, cache_dir=cdip_cache)

    # Cam-only diagnostic plot — always produced, useful even before CDIP catches up.
    cam_only_path = _save_cam_only(hs_df, tag)

    if buoy.empty:
        log.warning(
            "no CDIP %s data overlap; saved cam-only diagnostic to %s. "
            "CDIP typically lags realtime by ~6-12 h; re-run later.",
            site.buoy_id, cam_only_path,
        )
        return {
            "fit": None,
            "plots": {"cam_only": str(cam_only_path)},
            "n_paired": 0,
            "summary": None,
        }

    paired = validate.align_to_buoy(hs_df, buoy, cam_col="hs_px_4std")
    if len(paired) < 6:
        log.warning(
            "only %d paired samples; saved cam-only diagnostic. "
            "Re-run once more footage and buoy overlap accumulate.",
            len(paired),
        )
        return {
            "fit": None,
            "plots": {"cam_only": str(cam_only_path)},
            "n_paired": len(paired),
            "summary": None,
        }

    fit, paired = validate.fit_train_test(paired, train_frac=train_frac)
    plots = validate.plot_validation(paired, fit, tag=tag)

    site.calibration.scale_m_per_px = fit.scale_m_per_px
    site.calibration.bias_m = fit.bias_m
    site.calibration.fit_rmse_m = fit.rmse_test_m
    site.calibration.fit_n_samples = fit.n_train + fit.n_test
    site.calibration.fit_train_window = (
        paired.index[0].isoformat(),
        paired.index[fit.n_train - 1].isoformat(),
    )
    site.dump(site_path)

    summary_path = Path("reports") / f"validation_{tag}_summary.json"
    validate.save_summary(fit, site.calibration.fit_train_window, summary_path)

    return {
        "fit": fit,
        "plots": {k: str(v) for k, v in plots.items()},
        "n_paired": len(paired),
        "summary": str(summary_path),
    }
