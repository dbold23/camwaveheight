# CamWaveHeight — Build Plan

## STATUS NOTE (2026-05-25)
Pivoted off pier-piling calibration. The actual Scripps Pier cam (Surfline-routed,
publicly embedded on the SIO piercam page) is pier-mounted looking outward at
open ocean — no pilings visible. Replaced with **regression calibration**:
extract pixel-space Hs from a fixed surf-zone ROI and regress against CDIP 201
Hs to learn the meter scale. Closer to what Surfline themselves likely do
in production, since they can't bake site-specific homographies into every cam.



## Phase 1 MVP — Scripps Pier vs CDIP 073

Build order chosen to (a) start cam data accumulating ASAP and (b) test each module in isolation before wiring.

### 1. Project scaffolding
- [x] Repo + README + .gitignore
- [ ] `pyproject.toml` with deps pinned
- [ ] `src/camwaveheight/__init__.py`
- [ ] `configs/sites/scripps_pier.yaml` with cam URL, buoy ID, pier geometry, calibration placeholders
- [ ] `configs/site_schema.py` — pydantic schema so site YAMLs validate

### 2. CDIP buoy fetcher (`validate.py` — fetch half)
- [ ] `fetch_cdip_hs(station_id, start, end)` via erddapy → DataFrame of Hs/Tp/Tz at 30-min cadence
- [ ] Caching to `data/cdip/{station_id}_{start}_{end}.parquet`
- [ ] CLI: `python -m camwaveheight.validate fetch --station 073 --start 2026-05-01 --end 2026-05-25`
- [ ] Sanity plot saved to `reports/` — proves data shape and units

### 3. Ingest (`ingest.py`)
- [ ] Research Scripps Pier cam stream URL (likely HLS via SIO/CDIP)
- [ ] `record_stream(url, out_dir, segment_sec=600)` — ffmpeg subprocess, segmented MP4s with UTC timestamps in filename
- [ ] PTZ-event detection: log SSIM drop between segment-boundary frames; mark suspect segments
- [ ] CLI: `python -m camwaveheight.ingest --site scripps_pier --hours 72`

### 4. Calibration (`calibration.py`)
- [ ] `annotate_piling_bases(keyframe)` — interactive matplotlib clicker, saves 4–6 (u,v) pairs
- [ ] `solve_homography(pixel_pts, world_pts)` — uses known piling spacing (6m on-center)
- [ ] `pixel_to_meter(u, v)` callable
- [ ] Vertical scale: `m_per_px_at(u, v)` — derived from piling diameter (~1m) in pixels at that location
- [ ] Persist calibration to `configs/sites/scripps_pier.yaml`

### 5. Waterline detection (`waterline.py`)
- [ ] Per-frame: extract vertical pixel strip on selected piling
- [ ] Classical: HSV threshold + Canny + peak detection → row index of water/piling boundary
- [ ] Output: `eta_px_series(video_path, strip_x, y_range) → DataFrame(t, eta_px)`
- [ ] QC: drop frames where confidence < threshold; gap-fill short dropouts
- [ ] Decision point: if classical RMSE poor on a 5-min validation clip, train UNet on ~200 labeled frames

### 6. Stats (`stats.py`)
- [ ] `detrend(eta_t, fs, lowpass_sec=30)` — Butterworth lowpass for tide removal
- [ ] `scale_to_meters(eta_px, m_per_px)` — applies calibration
- [ ] `zero_crossing_hs(eta_m, window_min=17)` — H_1/3, Tz, Tp via rolling window
- [ ] FFT-based Tp as cross-check

### 7. Validation (`validate.py` — compare half)
- [ ] Align cam-derived Hs(t) with CDIP Hs(t) on common timestamps
- [ ] Metrics: RMSE, bias, scatter index, R²
- [ ] Plots: 1:1 scatter, time-series overlay, residuals vs swell period/direction
- [ ] Report builder: `reports/validation_v1.pdf`

### 8. Pipeline (`pipeline.py`)
- [ ] `run_site(config_path, t_start, t_end)` orchestrates ingest → waterline → stats → validation
- [ ] Idempotent: skips stages whose outputs exist

### Phase 1 success criterion
RMSE on Hs ≤ 25 cm vs CDIP 073 for sea states 0.5–2 m.

## Phase 2+ deferred until Phase 1 lands.

## Open decisions
- Stream URL for Scripps Pier cam — need to verify before ingest can run
- Whether to record locally or directly to external drive — default external drive given the 2TB available
- Classical waterline first vs train UNet immediately — defer until classical is benchmarked
