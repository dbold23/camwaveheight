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

    t0 = hs_df.index.min().floor("h")
    t1 = hs_df.index.max().ceil("h")
    buoy = cdip.fetch_cdip_params(site.buoy_id, t0, t1, cache_dir=cdip_cache)

    paired = validate.align_to_buoy(hs_df, buoy, cam_col="hs_px_4std")
    if len(paired) < 6:
        raise RuntimeError(
            f"only {len(paired)} paired samples after alignment; need ≥6. "
            f"Wait for more footage or relax window/step."
        )

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
