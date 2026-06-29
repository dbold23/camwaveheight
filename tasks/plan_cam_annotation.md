# Plan B — Cam analysis upgrade (Surfline-style per-frame annotation)

## Context

`camwaveheight` today is a single-signal pipeline: ffmpeg HLS recorder → per-frame **motion energy**
`mean(|I_t − I_{t-1}|)` in one fixed surf-zone ROI → rolling 17-min Hs/Tp → linear regression vs CDIP buoy 201.
It met Phase 1 (RMSE 7 cm random / 15.8 cm sequential vs the buoy) but produces only two scalar parquet tables
(`motion.parquet`, `eta_px.parquet`) over ~26 h of footage at Scripps Pier, La Jolla.

This plan turns the single-scalar pipeline into a multi-component per-frame annotation system matching a
Surfline-style screenshot (visibility/glare classification with probabilities; people/bird/surfer localization +
water/shore counts; breaking-wave segmentation + per-segment wave height; shoreline polyline + horizon line; PTZ
state; scale factor; frame/date), built on top of the validated motion-energy Hs, with a
foundation-model→distillation strategy. New `src/camwaveheight/perception/` subpackage.

### Conventions to reuse (verified)
- Frame timing: `ingest.segment_start_utc(path)` + `frame_idx/fps`; segments `data/raw/<site>/<YYYYmmdd>/seg-*.mp4`; `ingest.list_segments`.
- ROI `site.calibration.wave_roi` (x100,y420,w1080,h280); frame ~1280×720. Scale: `calibration_data/scale_marks.csv` = 36.2 px/m at row ~709 (board 1.9558m=70.8px); validated regression `scale_m_per_px=0.000956`, `bias_m=0.763`.
- Validation loop: `validate.align_to_buoy/fit_train_test/plot_validation/save_summary`, `cdip.fetch_cdip_params`, `stats.rolling_hs` (1020s/300s).
- Render style: `scripts/make_overlay_video.py` (mp4v VideoWriter, FONT_HERSHEY_SIMPLEX, yellow ROI, np.vstack panels, addWeighted heat).
- Env: Mac/MPS; `torch`/`ultralytics` in optional `ml` extra, not installed. All heavy imports lazy.

### Package layout
```
src/camwaveheight/perception/
  schema.py        # FrameAnnotation dataclass + parquet (de)serialize  <-- KEY DELIVERABLE
  frames.py        # iterate_frames(segment) using ingest timestamps
  detect.py        # detection + tracking (people/birds/surfers), water/shore zoning
  segment_waves.py # breaking-wave segmentation -> per-segment Hs
  shoreline.py     # shoreline polyline (water/sand)
  horizon.py       # horizon line (classical CV)
  visibility.py    # multi-class scene-quality classifier
  ptz.py           # PTZ change detection (SSIM at segment boundaries) + scale tracking
  height.py        # px<->m scale model (scale_marks.csv + per-row), per-wave Hs geometry
  teachers.py      # SAM2, GroundingDINO, YOLO-World, CLIP, Depth-Anything-v2 wrappers (lazy, device-aware)
  distill.py       # auto-label -> dataset export -> train student
  annotate.py      # unified per-frame orchestrator -> FrameAnnotation rows
  render.py        # single overlay renderer (extends make_overlay_video.py)
  reconcile.py     # compare per-wave / motion / whitewater features vs CDIP
  models/visibility_taxonomy.yaml, classes.yaml
```

### KEY DELIVERABLE — `schema.py` `FrameAnnotation` (one row/frame; variable-length fields as JSON strings; UTC `t` index, like eta_px.parquet)
Columns: `t`(idx), `segment`, `frame_idx`, `frame_number`, `date_str`, `fps`;
`visibility_label`, `visibility_top_class`, `visibility_probs`(JSON), `visibility_confidence`;
`count_total/water/shore/people/surfers/birds`, `detections`(JSON: cls,conf,box,zone,track_id);
`n_wave_segments`, `wave_segments`(JSON: mask_rle,bbox,crest_row,base_row,height_px,height_m,scale_px_per_m,conf),
`hs_per_wave_m_max`, `hs_per_wave_m_mean`, `whitewater_area_px`, `motion`, `v_mean`(carried from wave_detect);
`shoreline_polyline`(JSON), `shoreline_mean_row`, `horizon_y`, `horizon_slope`, `horizon_rgb`(JSON);
`ptz_state`, `ptz_pan/tilt/zoom`, `scale_factor`, `view_id`; `is_night`, `analysis_ok`.
Helpers: `to_parquet/read_parquet` (JSON cols parsed back), `EMPTY` factory, RLE encode/decode.
Output: `data/perception/<site>/annotations.parquet`.

### Dependencies (`pyproject.toml`)
- `perception = ["torch>=2.2","torchvision>=0.17","ultralytics>=8.3","supervision>=0.21","transformers>=4.44","pycocotools>=2.0","scikit-image>=0.24"]`
- `distill = ["transformers>=4.44","open-clip-torch>=2.24","timm>=1.0"]`
- SAM2 via `pip install git+https://github.com/facebookresearch/segment-anything-2.git`, gated behind try/except. GroundingDINO via transformers `IDEA-Research/grounding-dino-tiny` (no separate install).

### Phases
**P0 — zero-shot annotation overlay (fast visual win, no training).**
- `frames.py` (copy capture loop from `wave_detect.extract_motion_energy_series`).
- `teachers.py`: GroundingDINO (open-vocab `person.surfer.bird.dog.`), YOLO-World (`yolov8x-worldv2.pt`), SAM2, CLIPVisibility, DepthAnythingV2; device `mps→cuda→cpu`.
- `horizon.py` classical (Sobel/Hough upper-third → line + RGB tint). `shoreline.py` classical (color+texture water/sand boundary → polyline). `ptz.py` SSIM between last frame seg N / first frame seg N+1 → `view_id`.
- `height.py` `ScaleModel`: load scale_marks.csv anchor; `px_per_m_at_row(row)` linear-perspective (scale→0 at horizon); `height_m(crest_row,base_row)` integrate 1/px_per_m; also expose validated regression scale.
- `detect.py` zero-shot: teacher boxes → canonical classes (classes.yaml) → zone by box bottom-center vs shoreline → counts. `segment_waves.py` zero-shot: foam seed mask (reuse `wave_detect.detect_eta_in_frame` HSV logic) → connected comps → optional SAM2 refine → per-seg crest/base/height. `visibility.py` zero-shot: CLIP vs taxonomy text prompts.
- `annotate.py` orchestrates per frame (carry motion/v_mean; skip heavy models on night). `render.py` draws everything like the screenshot (boxes by class/zone+track id, cyan shoreline, RGB horizon, translucent wave masks w/ height labels, top-left visibility panel, top-right counts, footer frame#/date/PTZ/scale).
- `cwh annotate --site --clip --out --seconds --detector[grounding-dino|yolo-world|student] --sample-every`.
- **Verify:** `cwh annotate --clip data/test_clips/seg-20260526T025351Z.mp4 --out reports/annotated_demo.mp4 --seconds 20` → MP4 visibly matches screenshot layout; smoke test on 5 frames asserts full schema populated.

**P1 — per-wave Hs validated vs CDIP (measurement core).**
- Per-frame seg heights are instantaneous face heights, not Hs. Aggregate to rolling 17-min windows (reuse `stats.rolling_hs` cadence): mean of top-1/3 segment heights (`hs_seg_top3_m`) + whitewater-area stats.
- Feed chosen scalar through `validate.align_to_buoy`+`fit_train_test` exactly as today (as `cam_col`) → RMSE/R²/scatter vs CDIP 201.
- `reconcile.py`: align motion / per-wave top-1/3 / whitewater-area / eta_px on window grid; fit each to CDIP; report table; optional 2-feature blend (time-ordered split). Motion-energy stays system of record unless per-wave beats it on held-out RMSE.
- `segment_waves.per_wave_hs_for_site(...)` idempotent per-segment caching (mirror `extract_motion_for_site`) → `data/perception/<site>/wave_hs.parquet`.
- `site.Calibration` += optional `perwave_scale_m_per_px/bias_m/fit_rmse_m` (don't overwrite motion coeffs). `pipeline.py` optional `run_perception_hs` branch.
- `cwh perwave-hs --site --feature[per_wave|whitewater_area] --tag`.
- **Verify:** prints RMSE_test/R² vs CDIP; `reports/validation_perwave_v1_scatter.png`; `reconcile_v1.json` ranks features. Bar: RMSE_test ≤ 25 cm (motion baseline 7–16 cm).

**P2 — distillation: fast students + visibility classifier.**
- Loop (per component) in `distill.py`: teacher auto-labels recorded frames → export dataset → train student → eval agreement (+CDIP) → swap into annotate via `--detector student`.
- Detection: teacher GroundingDINO/YOLO-World pseudo-boxes → YOLO-format dataset → student **YOLO11n** + ByteTrack/BoT-SORT (bundled). Surfer = person whose track stays in water zone N frames.
- Wave seg: teacher SAM2 (prompted by foam-seed mask) → mask dataset → student small **UNet** (segmentation-models-pytorch, MobileNet encoder).
- Visibility: teacher **CLIP** soft probs → soft-target KL distillation into **MobileNetV3-Small** (timm) → calibrated per-class probabilities like the screenshot.
- Taxonomy `visibility_taxonomy.yaml`: fine {Clear,ClearGlare,HazyClear,ClearRainOrBlur,GlareClear,Glare,Hazy,RainOrBlur,Dark/Night} → coarse {Good,Fair,Poor,Night}, with CLIP prompts per class.
- **Hardware:** auto-label + train on GPU box (use `gpu-setup`/`env-check` skills), copy `models/*.pt` back; student inference runs on Mac/MPS at 1–5 fps sampling.
- `cwh autolabel --component[detect|waves|visibility] --max-frames --device`; `cwh distill --component --dataset --epochs --out`.
- **Verify:** students train; `reports/distill_visibility_eval.json` ≥0.85 top-1 vs CLIP; `cwh annotate --detector student` renders faster than P0.

**P3 — full unified parquet + dashboard.**
- `annotate.run_perception(site, ...)` over all segments, idempotent caching, PTZ `view_id` gating → `annotations.parquet`.
- `cwh perception-run --site --sample-every --detector --start --end`.
- `scripts/perception_dashboard.py` (streamlit, `dashboard` extra): time slider, counts/visibility timelines, per-wave Hs vs motion-Hs vs CDIP overlay, frame viewer using `render.py`.
- **Verify:** parquet has all schema columns ≥1 row/min; dashboard renders overlay + series.

### Non-negotiable: motion-energy pipeline stays system of record (keep its current RMSE), don't overwrite its site-YAML coeffs from perception; carry motion/v_mean into FrameAnnotation; per-wave Hs validated through the same `validate.py`.

### Execution order: schema/frames/height → horizon/shoreline/ptz → teachers+detect/segment/visibility+annotate+render+`cwh annotate` (P0) → per_wave_hs+reconcile+`cwh perwave-hs` (P1) → distill+students (P2) → run_perception+dashboard (P3).

### Cross-reference contract with Plan A (satellite): Plan A's `shoreline.py` exposes per-transect `cross_shore_position_m` on a UTC index. This plan's cam shoreline polyline can later be regressed against that absolute satellite-derived shoreline (same pattern as cam Hs → CDIP). Share transect IDs.

### Critical files: `wave_detect.py`, `cli.py`, `validate.py`, `scripts/make_overlay_video.py`, `pipeline.py` (+ refs: `site.py`, `stats.py`, `ingest.py`, `calibration_data/scale_marks.csv`).
