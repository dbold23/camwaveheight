"""Photogrammetric rectification for a fixed coastal camera.

Turns image pixels into metric coordinates on the sea surface, so wave signals
can be measured in meters from first principles — independent of the CDIP buoy
(which then becomes an *independent* validation rather than the calibration).

This is the Argus / coastal-imaging approach (Holman & Stanley 2007): recover
the camera pose from the horizon plus Ground Control Points (GCPs), then project
any pixel ray onto the tide-corrected water plane.

World frame (right-handed):
    X = across-track  (horizontal, parallel to shore)
    Y = range         (horizontal, positive out to sea, away from cam base)
    Z = up            (0 = mean water surface at calibration tide)
Camera is at (0, 0, H), optical axis depressed by `tilt` below horizontal,
looking out to sea (+Y), with `roll` about the optical axis.

The quantity we ultimately need is `vertical_scale_at_row(v)`: pixels per meter
of *elevation* at the sea surface imaged at row v. That converts a wave's pixel
excursion into meters. Note it differs from the horizontal (across-track) scale
by ~cos(depression angle) — near the camera the two differ by a few percent,
toward the horizon they converge. The full projection below handles this exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CameraParams:
    H: float          # camera height above water plane, meters
    f: float          # focal length, pixels
    tilt: float       # down-tilt of optical axis below horizontal, radians
    roll: float       # roll about optical axis, radians
    cx: float = 640.0
    cy: float = 360.0


def _rotation(tilt: float, roll: float) -> np.ndarray:
    """World->camera rotation. Rows are the camera axes (x=right, y=down, z=fwd)."""
    ct, st = np.cos(tilt), np.sin(tilt)
    z_cam = np.array([0.0, ct, -st])   # forward / optical axis
    x_cam = np.array([1.0, 0.0, 0.0])  # right (across-track)
    y_cam = np.cross(z_cam, x_cam)     # down
    R = np.vstack([x_cam, y_cam, z_cam])
    if roll:
        cr, sr = np.cos(roll), np.sin(roll)
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]])
        R = Rz @ R
    return R


def project(P: np.ndarray, p: CameraParams) -> np.ndarray:
    """World point(s) (N,3) -> pixel(s) (N,2)."""
    P = np.atleast_2d(P).astype(float)
    C = np.array([0.0, 0.0, p.H])
    R = _rotation(p.tilt, p.roll)
    Pc = (R @ (P - C).T).T
    u = p.cx + p.f * Pc[:, 0] / Pc[:, 2]
    v = p.cy + p.f * Pc[:, 1] / Pc[:, 2]
    return np.stack([u, v], axis=1)


def pixel_to_water(u: float, v: float, p: CameraParams) -> np.ndarray:
    """Pixel -> world point on the z=0 water plane (ray-plane intersection)."""
    R = _rotation(p.tilt, p.roll)
    C = np.array([0.0, 0.0, p.H])
    d_cam = np.array([(u - p.cx) / p.f, (v - p.cy) / p.f, 1.0])
    d_world = R.T @ d_cam
    if abs(d_world[2]) < 1e-9:
        return np.array([np.nan, np.nan, np.nan])
    t = -C[2] / d_world[2]
    return C + t * d_world


def horizon_row(p: CameraParams, u: float | None = None) -> float:
    """Image row of the horizon (rays to infinite range, Z=0)."""
    u = p.cx if u is None else u
    R = _rotation(p.tilt, p.roll)
    dirs = R @ np.array([0.0, 1.0, 0.0])  # direction toward sea at infinity
    return float(p.cy + p.f * dirs[1] / dirs[2])


def range_at_row(v: float, p: CameraParams) -> float:
    """Horizontal range (m) to the water point imaged at (cx, v)."""
    P = pixel_to_water(p.cx, v, p)
    return float(P[1])


def across_track_scale_at_row(v: float, p: CameraParams) -> float:
    """Pixels per meter, horizontal across-track, at the water point imaged at row v."""
    P = pixel_to_water(p.cx, v, p)
    a = project(P + np.array([0.5, 0, 0]), p)[0]
    b = project(P + np.array([-0.5, 0, 0]), p)[0]
    return float(np.hypot(*(a - b)))


def vertical_scale_at_row(v: float, p: CameraParams) -> float:
    """Pixels per meter of ELEVATION at the water point imaged at row v.

    This is the number that converts a wave's pixel excursion into meters.
    """
    P = pixel_to_water(p.cx, v, p)
    a = project(P + np.array([0, 0, 0.5]), p)[0]
    b = project(P + np.array([0, 0, -0.5]), p)[0]
    return float(np.hypot(*(a - b)))


# ---------------------------------------------------------------------------
# Horizon detection
# ---------------------------------------------------------------------------

def _band_drop_row(gray_band: np.ndarray, y0: int) -> float:
    """Row of steepest column-averaged brightness drop (sky->sea) in a band."""
    prof = gray_band.mean(axis=1)                       # average across columns
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 2).ravel()
    g = np.gradient(prof)                               # d(brightness)/d(row)
    return float(y0 + int(np.argmin(g)))               # steepest darkening


def detect_horizon(frame: np.ndarray, search=(260, 470)) -> tuple[float, float]:
    """Detect the sea/sky boundary. Returns (row_at_center, roll_radians).

    Column-AVERAGING first makes this cloud-robust: a real horizon darkens
    every column at the same row, so it dominates the averaged profile, whereas
    clouds average out. Roll is estimated from left- vs right-half horizon rows.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    H, W = gray.shape
    y0, y1 = search
    band = gray[y0:y1, :]
    row_center = _band_drop_row(band, y0)
    # roll from left/right thirds (avoid the very edges)
    row_L = _band_drop_row(gray[y0:y1, int(W * 0.15):int(W * 0.40)], y0)
    row_R = _band_drop_row(gray[y0:y1, int(W * 0.60):int(W * 0.85)], y0)
    slope = (row_R - row_L) / (W * 0.45)
    roll = float(np.arctan(slope))
    # guard against spurious roll from residual cloud structure
    if abs(np.degrees(roll)) > 5:
        log.warning("horizon roll %.1f deg looks large; clamping to 0", np.degrees(roll))
        roll = 0.0
    log.info("horizon: row@center=%.1f, roll=%.2f deg", row_center, np.degrees(roll))
    return float(row_center), roll


# ---------------------------------------------------------------------------
# Geometry fitting from horizon + GCPs
# ---------------------------------------------------------------------------

@dataclass
class ScaleGCP:
    """A known-length segment measured in the image.

    orientation: 'across' (horizontal, broadside/parallel to shore) or
    'vertical' (plumb, elevation). row is the image row where it sits.
    """
    row: float
    pixel_len: float
    length_m: float
    orientation: str  # 'across' | 'vertical'


def fit_geometry(
    horizon_row_obs: float,
    roll_obs: float,
    gcps: list[ScaleGCP],
    H_prior: float = 10.8,
    f_prior: float = 1100.0,
    cx: float = 640.0,
    cy: float = 360.0,
    H_sigma: float = 3.0,
    f_sigma: float = 400.0,
) -> tuple[CameraParams, dict]:
    """Solve camera H, f, tilt from the horizon row + scale GCPs (+ soft priors).

    With a single across-track GCP we have 2 hard constraints (horizon, scale)
    and 3 unknowns (H, f, tilt); the priors regularize the under-determination.
    More GCPs (vertical poles at several distances) over-determine it and let
    the priors go slack.
    """
    from scipy.optimize import least_squares

    def unpack(x):
        return CameraParams(H=x[0], f=x[1], tilt=x[2], roll=roll_obs, cx=cx, cy=cy)

    def residuals(x):
        p = unpack(x)
        res = [horizon_row(p) - horizon_row_obs]
        for g in gcps:
            if g.orientation == "across":
                pred = across_track_scale_at_row(g.row, p)
            else:
                pred = vertical_scale_at_row(g.row, p)
            obs = g.pixel_len / g.length_m
            res.append(pred - obs)
        # soft priors
        res.append((x[0] - H_prior) / H_sigma)
        res.append((x[1] - f_prior) / f_sigma)
        return res

    x0 = [H_prior, f_prior, np.radians(2.0)]
    sol = least_squares(residuals, x0, method="lm", max_nfev=5000)
    p = unpack(sol.x)
    info = {
        "success": sol.success,
        "cost": float(sol.cost),
        "horizon_pred": horizon_row(p),
        "horizon_obs": horizon_row_obs,
        "tilt_deg": np.degrees(p.tilt),
    }
    for g in gcps:
        pred = (across_track_scale_at_row if g.orientation == "across"
                else vertical_scale_at_row)(g.row, p)
        info[f"gcp_row{int(g.row)}_{g.orientation}"] = {
            "obs_px_per_m": round(g.pixel_len / g.length_m, 2),
            "pred_px_per_m": round(pred, 2),
        }
    return p, info
