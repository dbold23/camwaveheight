"""Fit the coastal-camera geometry from the horizon + field GCPs, and report.

Reads calibration_data/scale_marks.csv, detects the horizon in a daytime frame,
solves the camera pose, and writes a diagnostic figure + the vertical-scale
curve used to convert wave pixel excursions into meters.
"""

from __future__ import annotations

import argparse
import csv
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

from camwaveheight import rectify

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_gcps(path: str) -> list[rectify.ScaleGCP]:
    gcps = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if not r.get("length_m"):
                continue
            orient = "vertical" if "vert" in r["label"].lower() else "across"
            gcps.append(rectify.ScaleGCP(
                row=float(r["mean_row"]), pixel_len=float(r["pixel_len"]),
                length_m=float(r["length_m"]), orientation=orient))
    return gcps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default="/tmp/daytime_horizon.png")
    ap.add_argument("--gcps", default="calibration_data/scale_marks.csv")
    ap.add_argument("--surf-rows", default="480,620", help="row range where waves break")
    ap.add_argument("--out", default="reports/rectify_report.png")
    args = ap.parse_args()

    fr = cv2.imread(args.frame)
    v_h, roll = rectify.detect_horizon(fr)
    gcps = load_gcps(args.gcps)
    print(f"loaded {len(gcps)} GCP(s): " + ", ".join(
        f"row{int(g.row)}/{g.orientation}/{g.pixel_len/g.length_m:.1f}px·m⁻¹" for g in gcps))

    p, info = rectify.fit_geometry(v_h, roll, gcps)
    print(f"\nH={p.H:.2f} m  f={p.f:.0f} px  tilt={np.degrees(p.tilt):.2f}°  "
          f"FOV={2*np.degrees(np.arctan(p.cx/p.f)):.0f}°")
    for k, val in info.items():
        if k.startswith("gcp"):
            print(f"  {k}: {val}")

    rows = np.arange(400, 715, 5)
    vscale = np.array([rectify.vertical_scale_at_row(r, p) for r in rows])
    ranges = np.array([rectify.range_at_row(r, p) for r in rows])

    s0, s1 = (int(x) for x in args.surf_rows.split(","))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # (1) frame with horizon + surf band + GCPs
    disp = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).copy()
    ax[0].imshow(disp)
    ax[0].axhline(v_h, color="cyan", lw=1.5, label=f"horizon (row {v_h:.0f})")
    ax[0].axhspan(s0, s1, color="yellow", alpha=0.15, label="surf zone")
    for g in gcps:
        ax[0].axhline(g.row, color="red", lw=1, ls=":")
    ax[0].set_title(f"pose: H={p.H:.1f}m  f={p.f:.0f}px  tilt={np.degrees(p.tilt):.1f}°")
    ax[0].legend(loc="lower right", fontsize=8)
    ax[0].axis("off")
    # (2) vertical scale vs row
    ax[1].plot(vscale, rows, color="C0")
    ax[1].axhspan(s0, s1, color="yellow", alpha=0.2)
    for g in gcps:
        ax[1].scatter(g.pixel_len / g.length_m, g.row, color="red", zorder=5,
                      label=f"GCP row{int(g.row)}")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("vertical scale (px per meter)")
    ax[1].set_ylabel("image row")
    ax[1].set_title("vertical scale vs row")
    ax[1].legend(fontsize=8)
    # (3) range vs row
    ax[2].plot(ranges, rows, color="C2")
    ax[2].axhspan(s0, s1, color="yellow", alpha=0.2)
    ax[2].invert_yaxis()
    ax[2].set_xlabel("horizontal range (m)")
    ax[2].set_ylabel("image row")
    ax[2].set_title("range to sea surface vs row")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nwrote {args.out}")

    surf_v = [rectify.vertical_scale_at_row(r, p) for r in range(s0, s1 + 1, 10)]
    print(f"surf-zone vertical scale: {min(surf_v):.1f}-{max(surf_v):.1f} px/m "
          f"(1 px = {100/max(surf_v):.1f}-{100/min(surf_v):.1f} cm)")


if __name__ == "__main__":
    main()
