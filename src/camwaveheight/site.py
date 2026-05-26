"""Site config schema. Each beach is a YAML that conforms to `Site`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class PierGeometry(BaseModel):
    piling_spacing_m: float = Field(..., description="On-center spacing between pilings, meters.")
    piling_diameter_m: float = Field(..., description="Piling diameter, meters.")
    deck_elev_m_mllw: float | None = Field(None, description="Deck elevation above MLLW.")


class WaveROI(BaseModel):
    """Image-space region of interest for the surf zone."""

    x: int  # left
    y: int  # top
    w: int
    h: int


class Calibration(BaseModel):
    """Filled in by calibration step. Regression-mode is the default for cams
    with no metric reference in frame: we fit pixel-Hs to CDIP-Hs and store
    the linear coefficients here."""

    keyframe_path: str | None = None
    wave_roi: WaveROI | None = None
    # Regression-mode coefficients: meter_hs = scale * pixel_hs + bias
    scale_m_per_px: float | None = None
    bias_m: float | None = None
    fit_rmse_m: float | None = None
    fit_n_samples: int | None = None
    fit_train_window: tuple[str, str] | None = None  # ISO UTC start, end


class Site(BaseModel):
    name: str
    cam_url: str | None = None  # may be None if using local-only recordings
    cam_referer: str | None = None  # HTTP Referer header for CDN-protected streams
    calibration_method: Literal["pier_piling", "stationary_ref", "surfer_pose", "regression"] = (
        "regression"
    )
    buoy_id: str = Field(..., description="CDIP station number, e.g. '073'.")
    pier_geometry: PierGeometry | None = None
    calibration: Calibration = Calibration()
    notes: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Site":
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)
