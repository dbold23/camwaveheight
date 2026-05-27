"""Pixel-space wave-surface-elevation detector.

For a surf-zone ROI, per-frame find the topmost row whose pixel-luminance
exceeds a foam threshold. That row index is our η_px(t) — lower row number
means higher wave crest in the image. The time series, sampled at video
framerate, drives the rolling-window Hs computation downstream.

Why this works on a horizon-pointing cam without metric calibration:
- Breaking waves leave bright foam at their crest.
- In a fixed ROI the highest foam row tracks wave-crest height in pixels.
- Surface elevation in pixels is monotonic with surface elevation in meters
  for a fixed line of sight; the meter scale is fit later via regression
  against CDIP Hs.

Trade-offs:
- Requires foam (i.e. breaking waves). Glassy non-breaking conditions blank
  the signal; flag those windows and skip them.
- Sensitive to lens flare, sky reflection, surfer wakes. We filter by
  saturation (foam is desaturated) and a sky mask above the ROI.
- Daylight only. At night, foam is invisible — `confidence_mask` rejects
  those frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from camwaveheight.ingest import segment_start_utc
from camwaveheight.site import WaveROI

log = logging.getLogger(__name__)

DEFAULT_FOAM_V_MIN = 180  # floor for absolute V threshold (foam never dimmer than this)
DEFAULT_FOAM_S_MAX = 40   # foam is achromatic; sun glitter is more saturated. Strict.
DEFAULT_MIN_FOAM_PX = 25  # minimum foam pixels in a row to count it as "wave detected"
DEFAULT_VERTICAL_COHERENCE = 2  # row must have a foam-rich neighbor to count
DEFAULT_ADAPTIVE_K = 2.0  # adaptive_thresh = mean(V) + k * std(V); rejects "bright outliers"
DEFAULT_DARK_MEAN_V = 70   # ROI mean V below this = cam can't see waves; frame is "dark"


@dataclass
class FrameSample:
    t: pd.Timestamp
    eta_px: float | np.floating  # row index of topmost foam, np.nan if none detected
    foam_count: int


def detect_eta_in_frame(
    frame: np.ndarray,
    roi: WaveROI,
    foam_v_min: int = DEFAULT_FOAM_V_MIN,
    foam_s_max: int = DEFAULT_FOAM_S_MAX,
    min_foam_px: int = DEFAULT_MIN_FOAM_PX,
    vertical_coherence: int = DEFAULT_VERTICAL_COHERENCE,
    adaptive_k: float = DEFAULT_ADAPTIVE_K,
) -> tuple[float, int]:
    """Find the topmost row in `roi` of a vertically coherent foam band.

    A row counts only if at least `vertical_coherence` consecutive rows
    (including itself) each have >= min_foam_px foam pixels. This rejects
    thin specular reflections (horizon glare, sun on water) which appear
    in a single row only, while keeping wave crests (always span several
    rows vertically because the wave face is tall).

    Returns:
        (eta_px, foam_count): eta_px in original frame coordinates
        (NaN if no coherent foam band); foam_count is total foam pixels in ROI.
    """
    x0, y0, w, h = roi.x, roi.y, roi.w, roi.h
    crop = frame[y0 : y0 + h, x0 : x0 + w]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    V, S = hsv[:, :, 2], hsv[:, :, 1]
    # Night filter: if the ROI is too dark, cam is functionally blind.
    if V.mean() < DEFAULT_DARK_MEAN_V:
        return float("nan"), 0
    # Adaptive V threshold per frame, floored at the absolute foam_v_min.
    # `mean + k*std` keeps only "outlier-bright" pixels — works across lighting.
    adaptive_thresh = max(int(V.mean() + adaptive_k * V.std()), foam_v_min)
    foam_mask = (V >= adaptive_thresh) & (S <= foam_s_max)
    per_row_count = foam_mask.sum(axis=1)
    foam_total = int(foam_mask.sum())

    rich = per_row_count >= min_foam_px
    if vertical_coherence > 1:
        # A row is "coherent" if it AND the next (coherence-1) rows are all rich.
        coherent = np.copy(rich)
        for k in range(1, vertical_coherence):
            shifted = np.r_[rich[k:], np.zeros(k, dtype=bool)]
            coherent &= shifted
    else:
        coherent = rich

    rows = np.where(coherent)[0]
    if rows.size == 0:
        return float("nan"), foam_total
    return float(y0 + rows[0]), foam_total


def extract_eta_series(
    video_path: str | Path,
    roi: WaveROI,
    sample_every_n_frames: int = 1,
    foam_v_min: int = DEFAULT_FOAM_V_MIN,
    foam_s_max: int = DEFAULT_FOAM_S_MAX,
    min_foam_px: int = DEFAULT_MIN_FOAM_PX,
) -> pd.DataFrame:
    """Run the detector across all frames in a segment.

    The segment's UTC start time is parsed from its filename; frame timestamps
    are `start + frame_index / fps`.

    Args:
        video_path: path to an MP4 segment (filename must encode UTC start).
        roi: image-space surf-zone ROI.
        sample_every_n_frames: stride; e.g. 3 → ~10 Hz from 30 fps video.
            Wave periods are 6–20 s, so 5 Hz is ample.

    Returns:
        DataFrame indexed by UTC timestamp with columns `eta_px`, `foam_count`.
    """
    path = Path(video_path)
    start = pd.Timestamp(segment_start_utc(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    rows: list[FrameSample] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every_n_frames == 0:
                eta_px, foam = detect_eta_in_frame(
                    frame, roi, foam_v_min, foam_s_max, min_foam_px
                )
                rows.append(
                    FrameSample(
                        t=start + pd.Timedelta(seconds=frame_idx / fps),
                        eta_px=eta_px,
                        foam_count=foam,
                    )
                )
            frame_idx += 1
    finally:
        cap.release()

    df = pd.DataFrame([r.__dict__ for r in rows]).set_index("t")
    df.index = pd.to_datetime(df.index, utc=True)
    log.info(
        "%s: %d samples, %.1f%% with foam detected",
        path.name,
        len(df),
        100 * df["eta_px"].notna().mean(),
    )
    return df


def extract_motion_energy_series(
    video_path: str | Path,
    roi: WaveROI,
    sample_every_n_frames: int = 6,
) -> pd.DataFrame:
    """Per-frame "wave motion energy" = mean(|I_t - I_{t-1}|) in the ROI.

    Lighting-invariant: frame differencing cancels static lighting and slow
    drift. The remaining signal is dominated by moving foam and water surface
    deformation — both of which scale with wave activity.

    Also returns ROI brightness mean (V channel) so night frames can be
    masked downstream.
    """
    path = Path(video_path)
    start = pd.Timestamp(segment_start_utc(path))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    prev_gray = None
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_n_frames == 0:
            crop = frame[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
            v_mean = float(gray.mean())
            if prev_gray is not None:
                motion = float(np.abs(gray - prev_gray).mean())
            else:
                motion = float("nan")
            rows.append({
                "t": start + pd.Timedelta(seconds=frame_idx / fps),
                "motion": motion,
                "v_mean": v_mean,
            })
            prev_gray = gray
        frame_idx += 1
    cap.release()

    df = pd.DataFrame(rows).set_index("t")
    df.index = pd.to_datetime(df.index, utc=True)
    # Mask night frames
    df.loc[df["v_mean"] < DEFAULT_DARK_MEAN_V, "motion"] = float("nan")
    log.info(
        "%s: %d frames; motion mean=%.2f (non-night: %.2f)",
        path.name, len(df), df["motion"].mean(), df.dropna()["motion"].mean(),
    )
    return df


def extract_motion_for_site(
    site_name: str,
    roi: WaveROI,
    out_root: str | Path = "data/raw",
    cache_path: str | Path | None = "data/eta/motion.parquet",
    sample_every_n_frames: int = 6,
) -> pd.DataFrame:
    """Run motion extraction over every segment for a site."""
    from camwaveheight.ingest import list_segments

    segs = list_segments(site_name, out_root=out_root)
    if not segs:
        raise FileNotFoundError(f"no segments under {out_root}/{site_name}")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing_t: set[pd.Timestamp] = set()
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            existing_t = set(cached.index.floor("min"))
        else:
            cached = pd.DataFrame()
    else:
        cached = pd.DataFrame()
        existing_t = set()

    new_frames: list[pd.DataFrame] = []
    for seg in segs:
        t0 = pd.Timestamp(segment_start_utc(seg)).floor("min")
        if t0 in existing_t:
            continue
        try:
            df = extract_motion_energy_series(seg, roi, sample_every_n_frames)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", seg.name, exc)
            continue
        new_frames.append(df)

    if not new_frames and cached.empty:
        return cached
    combined = pd.concat([cached, *new_frames]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    if cache_path is not None:
        combined.to_parquet(cache_path)
        log.info("wrote %s (%d rows)", cache_path, len(combined))
    return combined


def extract_eta_for_site(
    site_name: str,
    roi: WaveROI,
    out_root: str | Path = "data/raw",
    cache_path: str | Path | None = "data/eta/scripps_pier_eta_px.parquet",
    sample_every_n_frames: int = 3,
) -> pd.DataFrame:
    """Run extraction over every segment for a site and concatenate."""
    from camwaveheight.ingest import list_segments

    segs = list_segments(site_name, out_root=out_root)
    if not segs:
        raise FileNotFoundError(f"no segments under {out_root}/{site_name}")

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing_t: set[pd.Timestamp] = set()
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            # We re-derive per segment for idempotency: filter out segments
            # whose first sample is already in the cache.
            existing_t = set(cached.index.floor("min"))
        else:
            cached = pd.DataFrame()
    else:
        cached = pd.DataFrame()
        existing_t = set()

    new_frames: list[pd.DataFrame] = []
    for seg in segs:
        t0 = pd.Timestamp(segment_start_utc(seg)).floor("min")
        if t0 in existing_t:
            continue
        try:
            df = extract_eta_series(seg, roi, sample_every_n_frames=sample_every_n_frames)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", seg.name, exc)
            continue
        new_frames.append(df)

    if not new_frames and cached.empty:
        return cached
    combined = pd.concat([cached, *new_frames]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    if cache_path is not None:
        combined.to_parquet(cache_path)
        log.info("wrote %s (%d rows)", cache_path, len(combined))
    return combined
