"""Show the foam mask overlaid on a frame at different thresholds."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camwaveheight.site import Site


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="reports/foam_mask_viz.png")
    ap.add_argument("--frame", type=int, default=300)
    args = ap.parse_args()

    site = Site.load(args.site)
    roi = site.calibration.wave_roi
    cap = cv2.VideoCapture(args.clip)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    assert ok

    crop = frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]

    thresholds = [(170, 90), (200, 50), (220, 30), (235, 20)]
    n = len(thresholds) + 2

    fig, axes = plt.subplots(2, n // 2 + 1, figsize=(3*(n//2+1), 4.5))
    axes = axes.flatten()

    axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title("ROI crop")
    axes[0].axis("off")

    axes[1].imshow(V, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("V (brightness)")
    axes[1].axis("off")

    for i, (vmin, smax) in enumerate(thresholds):
        mask = (V >= vmin) & (S <= smax)
        ax = axes[2 + i]
        ax.imshow(mask, cmap="gray")
        per_row = mask.sum(axis=1)
        # Find topmost row with > 25 foam pixels
        top = np.where(per_row >= 25)[0]
        top_y = (roi.y + top[0]) if top.size else None
        ax.set_title(f"V≥{vmin}, S≤{smax}\nfoam={mask.sum()}, top_y={top_y}")
        ax.axis("off")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
