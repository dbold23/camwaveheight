"""Render an annotated video showing the measurement in action.

For each frame:
  - draw the surf-zone ROI box
  - overlay the inter-frame motion as a red heat layer inside the ROI
  - plot the live motion-energy trace scrolling at the bottom

Output is an MP4 you can scrub to see the pipeline track wave activity.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from camwaveheight.site import Site
from camwaveheight.wave_detect import DEFAULT_DARK_MEAN_V


def draw_trace(canvas_w: int, h: int, history: deque, vmax: float) -> np.ndarray:
    """Render the scrolling motion-energy trace as a BGR strip."""
    strip = np.full((h, canvas_w, 3), 25, np.uint8)
    if len(history) < 2:
        return strip
    pts = list(history)
    n = len(pts)
    xs = np.linspace(0, canvas_w - 1, n).astype(int)
    ys = (h - 10 - (np.array(pts) / max(vmax, 1e-6)) * (h - 25)).clip(5, h - 5).astype(int)
    for i in range(1, n):
        cv2.line(strip, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (80, 140, 255), 2)
    cv2.putText(strip, "motion energy (wave activity)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="reports/overlay_demo.mp4")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--trace-h", type=int, default=160)
    args = ap.parse_args()

    site = Site.load(args.site)
    roi = site.calibration.wave_roi
    assert roi is not None

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(args.seconds * fps)

    out_h = H + args.trace_h
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (W, out_h))

    history: deque = deque(maxlen=int(fps * 12))  # 12s scrolling window
    prev_gray = None
    # First pass over the window to fix a stable vmax for the trace axis.
    motions = []
    idx = 0
    cap2 = cv2.VideoCapture(args.clip)
    while idx < max_frames:
        ok, frame = cap2.read()
        if not ok:
            break
        crop = frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if prev_gray is not None and g.mean() >= DEFAULT_DARK_MEAN_V:
            motions.append(float(np.abs(g - prev_gray).mean()))
        prev_gray = g
        idx += 1
    cap2.release()
    vmax = float(np.percentile(motions, 98)) if motions else 30.0

    prev_gray = None
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)

        if prev_gray is not None:
            diff = np.abs(g - prev_gray).astype(np.uint8)
            motion_val = float(diff.mean())
            # Heat overlay: red where motion is high
            heat = cv2.applyColorMap((diff * 4).clip(0, 255).astype(np.uint8), cv2.COLORMAP_HOT)
            blended = cv2.addWeighted(frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w], 0.6, heat, 0.4, 0)
            frame[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w] = blended
            history.append(motion_val)
        prev_gray = g

        # ROI box + label
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), (0, 255, 255), 2)
        cv2.putText(frame, "surf-zone ROI  |  motion-energy heat overlay", (roi.x, roi.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if history:
            cv2.putText(frame, f"motion = {history[-1]:5.1f}", (roi.x, roi.y + roi.h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 140, 255), 2, cv2.LINE_AA)

        trace = draw_trace(W, args.trace_h, history, vmax)
        canvas = np.vstack([frame, trace])
        writer.write(canvas)
        idx += 1

    cap.release()
    writer.release()
    print(f"wrote {args.out}  ({idx} frames, vmax={vmax:.1f})")


if __name__ == "__main__":
    main()
