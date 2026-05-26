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


class Calibration(BaseModel):
    """Filled in by `calibration.py` after annotation."""

    homography: list[list[float]] | None = None  # 3x3 row-major
    keyframe_path: str | None = None
    piling_strip_x: int | None = None  # column index of the selected piling
    piling_strip_y_range: tuple[int, int] | None = None  # vertical bounds of strip
    m_per_px_vertical: float | None = None
    annotated_pixel_points: list[tuple[float, float]] | None = None
    world_points_m: list[tuple[float, float]] | None = None


class Site(BaseModel):
    name: str
    cam_url: str | None = None  # may be None if using local-only recordings
    calibration_method: Literal["pier_piling", "stationary_ref", "surfer_pose"] = "pier_piling"
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
