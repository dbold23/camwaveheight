"""Sweep foam-detection thresholds across daytime + nighttime clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from camwaveheight.site import Site
from camwaveheight.wave_detect import extract_eta_series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    args = ap.parse_args()

    site = Site.load(args.site)
    roi = site.calibration.wave_roi
    assert roi is not None

    rows = []
    for v_min in [170, 200, 220, 230, 240]:
        for s_max in [30, 50, 80]:
            for min_px in [10, 25, 50]:
                for clip in args.clips:
                    df = extract_eta_series(
                        clip, roi, sample_every_n_frames=15,
                        foam_v_min=v_min, foam_s_max=s_max,
                        min_foam_px=min_px,
                    )
                    eta = df["eta_px"].dropna()
                    pinned_at_top = (eta == roi.y).mean() if len(eta) > 0 else 0
                    rows.append({
                        "clip": Path(clip).stem,
                        "v_min": v_min,
                        "s_max": s_max,
                        "min_px": min_px,
                        "det_rate": df["eta_px"].notna().mean(),
                        "eta_std": eta.std() if len(eta) > 10 else float("nan"),
                        "eta_median": eta.median() if len(eta) > 0 else float("nan"),
                        "pinned_top": pinned_at_top,
                        "n": len(df),
                    })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("reports/tune_sweep.csv", index=False)


if __name__ == "__main__":
    main()
