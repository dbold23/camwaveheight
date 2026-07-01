"""Mark a known-length object in a frame to calibrate pixel<->meter scale.

Extracts the exact frame at a given timecode, opens it, and lets you click the
two endpoints of a known-length object (e.g. a surfboard). Records the pixel
length, image location, and pixels-per-meter; appends to a CSV so multiple
calibration positions accumulate.

Usage:
    .venv/bin/python scripts/mark_scale.py \
        --clip data/raw/scripps_pier/20260529/seg-20260529T030130Z.mp4 \
        --sec 431 --length-m 1.88 --label board_pos1

Then click END 1, then END 2 of the object. Close the window.

If the interactive window won't open, pass the pixel coords directly:
    ... --p1 1190,past --p2 1240,705   (x,y pairs read from Preview)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

CSV_PATH = Path("calibration_data/scale_marks.csv")
FIELDS = ["label", "clip", "sec", "x1", "y1", "x2", "y2",
          "pixel_len", "length_m", "px_per_m", "mean_row", "note"]


def get_frame(clip: str, sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {clip} @ {sec}s")
    return fr


def click_two_points(
    frame_rgb: np.ndarray, sec: float, extent: tuple | None = None
) -> list[tuple[float, float]]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 10))
    # extent maps the (possibly cropped) display back to full-frame pixel coords,
    # so ginput returns coordinates in the ORIGINAL image space regardless of crop/zoom.
    ax.imshow(frame_rgb, extent=extent, origin="upper")
    ax.set_title(f"t={sec:.1f}s  —  click END 1, then END 2 of the object  (then close)")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--sec", type=float, required=True)
    ap.add_argument("--length-m", type=float, default=None,
                    help="True length of the object between the two clicked ends, meters.")
    ap.add_argument("--label", required=True)
    ap.add_argument("--p1", default=None, help="Bypass clicker: 'x,y' of end 1.")
    ap.add_argument("--p2", default=None, help="Bypass clicker: 'x,y' of end 2.")
    ap.add_argument("--note", default="")
    ap.add_argument("--brighten", type=float, default=1.0, help="Display brightness multiplier.")
    ap.add_argument("--no-enhance", action="store_true",
                    help="Show the raw frame with no brightness/CLAHE (clicks are identical either way).")
    ap.add_argument("--crop", default=None,
                    help="Zoom to a region 'x0,y0,x1,y1' (full-frame px). Clicks map back to full frame. "
                         "Preset 'br' = bottom-right.")
    args = ap.parse_args()

    fr = get_frame(args.clip, args.sec)
    disp = fr.copy()
    if not args.no_enhance:
        if args.brighten != 1.0:
            disp = cv2.convertScaleAbs(disp, alpha=args.brighten, beta=10)
        # adaptive enhance for visibility
        lab = cv2.cvtColor(disp, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        disp = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    full_disp = disp.copy()  # keep full frame for annotation (clicks are in full-frame coords)
    H, W = disp.shape[:2]
    extent = None
    if args.crop:
        if args.crop == "br":
            cx0, cy0, cx1, cy1 = int(W * 0.55), int(H * 0.62), W, H
        else:
            cx0, cy0, cx1, cy1 = (int(v) for v in args.crop.split(","))
        disp = disp[cy0:cy1, cx0:cx1]
        extent = (cx0, cx1, cy1, cy0)  # (left,right,bottom,top) -> clicks return full-frame px
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

    if args.p1 and args.p2:
        x1, y1 = map(float, args.p1.split(","))
        x2, y2 = map(float, args.p2.split(","))
    else:
        pts = click_two_points(rgb, args.sec, extent=extent)
        if len(pts) < 2:
            raise SystemExit("need two clicks; aborted")
        (x1, y1), (x2, y2) = pts

    pixel_len = float(np.hypot(x2 - x1, y2 - y1))
    mean_row = (y1 + y2) / 2
    px_per_m = pixel_len / args.length_m if args.length_m else None

    # Save an annotated confirmation image (on the full frame; coords are full-frame)
    ann = full_disp.copy()
    cv2.line(ann, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for (x, y) in [(x1, y1), (x2, y2)]:
        cv2.circle(ann, (int(x), int(y)), 6, (0, 255, 255), 2)
    txt = f"{args.label}: {pixel_len:.1f}px"
    if px_per_m:
        txt += f" = {args.length_m}m -> {px_per_m:.1f}px/m"
    cv2.putText(ann, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    Path("calibration_data").mkdir(exist_ok=True)
    ann_path = f"calibration_data/mark_{args.label}.png"
    cv2.imwrite(ann_path, ann)

    # Append to CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "label": args.label, "clip": args.clip, "sec": args.sec,
            "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
            "pixel_len": round(pixel_len, 1), "length_m": args.length_m,
            "px_per_m": round(px_per_m, 2) if px_per_m else "",
            "mean_row": round(mean_row, 1), "note": args.note,
        })

    print(f"[{args.label}] pixel_len={pixel_len:.1f}px  mean_row={mean_row:.0f}")
    if px_per_m:
        print(f"  -> {px_per_m:.1f} px/m at this location")
    print(f"  annotated: {ann_path}")
    print(f"  appended to {CSV_PATH}")


if __name__ == "__main__":
    main()
