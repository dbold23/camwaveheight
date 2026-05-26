"""Run the wave detector on a single clip and overlay η_px on a strip of frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camwaveheight.site import Site
from camwaveheight.wave_detect import detect_eta_in_frame, extract_eta_series


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="reports/detect_preview.png")
    ap.add_argument("--n-frames", type=int, default=6)
    ap.add_argument("--foam-v-min", type=int, default=180)
    ap.add_argument("--foam-s-max", type=int, default=80)
    args = ap.parse_args()

    site = Site.load(args.site)
    roi = site.calibration.wave_roi
    assert roi is not None, "set ROI first via `cwh set-roi`"

    df = extract_eta_series(
        args.clip, roi, sample_every_n_frames=3,
        foam_v_min=args.foam_v_min, foam_s_max=args.foam_s_max,
    )
    print(f"samples: {len(df)}; foam-detected: {df['eta_px'].notna().mean():.1%}")
    print(df.describe())

    # Overlay η on a strip of evenly spaced frames.
    cap = cv2.VideoCapture(args.clip)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, nframes - 1, args.n_frames).astype(int)
    fig, axes = plt.subplots(1, args.n_frames, figsize=(3 * args.n_frames, 2.5))
    for ax, idx in zip(axes, idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        eta_px, foam = detect_eta_in_frame(
            frame, roi,
            foam_v_min=args.foam_v_min, foam_s_max=args.foam_s_max,
        )
        # Box ROI
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), (0, 255, 255), 2)
        if eta_px == eta_px:  # not nan
            cv2.line(frame, (roi.x, int(eta_px)), (roi.x + roi.w, int(eta_px)), (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {idx}\nη_px={eta_px:.0f}, foam={foam}")
        ax.axis("off")
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}")

    # Also save the eta series plot
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(df.index, df["eta_px"], ".", markersize=3)
    ax2.invert_yaxis()  # lower row = higher in image
    ax2.set_ylabel("η_px (image row; lower = higher wave)")
    ax2.set_title(f"η_px(t) — {Path(args.clip).name}")
    eta_out = args.out.replace(".png", "_eta.png")
    fig2.tight_layout()
    fig2.savefig(eta_out, dpi=140)
    print(f"wrote {eta_out}")


if __name__ == "__main__":
    main()
