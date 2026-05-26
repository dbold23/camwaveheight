# CamWaveHeight — Webcam-Based Wave Height Inference

A CV pipeline that estimates significant wave height (Hs) at California beaches from public webcam feeds, calibrated via pier geometry and validated against CDIP nearshore buoys.

## Positioning

Surfline runs CV-driven wave-height inference across their cam network operationally. The exact method is closed. This project reverse-engineers the problem using public data, produces a quantitative validation against CDIP ground truth, and demonstrates an end-to-end pipeline relevant to their forecast operations team. Pitch in the cover letter as: *"a self-built validator that quantifies forecast accuracy at the beach scale."*

## System overview

```
[Webcam feed] → [Calibration (pixels↔meters)] → [Wave detection (η pixel)]
                                                       ↓
[CDIP buoy Hs] ← [Validation] ← [Wave statistics (Hs, Tp, Tz)] ← [η(t) meters]
```

## Data sources (use these, not Surfline cams)

Skip Surfline cams — their ToS prohibits scraping/redistribution. Use:

- **Scripps Pier cam** — operated by Scripps Institution of Oceanography / CDIP. Research-friendly, co-located with CDIP 201 (Scripps Nearshore buoy). This is the MVP site.
- **Explore.org cams** — Torrey Pines, Trestles area, some NorCal spots. Publicly streamed, generally permissive for research demos.
- **NOAA NOS webcams** — for select harbor/coastal sites.
- **Your own recordings** — phone tripod at a known beach with known scale objects. Worst case fallback that's 100% ToS-clean.

Ground truth: **CDIP THREDDS / ERDDAP** for nearshore buoy spectra. Use `erddapy` to pull Hs at 30-min cadence co-located with the cam.

## Phase 1 — MVP at Scripps Pier (target: 1–2 weeks)

Single-site proof of concept. The pier is the easiest case because piling geometry gives a clean stationary scale right in the swash zone.

**Known geometry to bake in:**
- Pier pilings ~6 m on-center, ~1 m diameter (verify in pier engineering docs — Scripps facilities should have these)
- Pier deck elevation ~7 m above MLLW
- Closest CDIP buoy: CDIP 201 (Scripps Nearshore) — confirm station ID before you start

**Pipeline steps for the agents:**

1. **Ingest** — record 48–72h of Scripps Pier cam footage at native framerate. Store as segmented MP4s with timestamps. Note any PTZ/zoom changes — those break calibration.

2. **Calibration module** (`calibration.py`)
   - Manually annotate 4–6 piling base points in a keyframe
   - Use known piling spacing to solve for ground-plane homography (perspective rectification)
   - Output: `pixel_to_meter(x, y)` function valid for the imaged water surface, plus a vertical-pixels-per-meter estimate along each piling

3. **Waterline detection** (`waterline.py`)
   - For one selected piling, extract a vertical pixel strip
   - Detect water/piling boundary per frame. Start with classical: HSV thresholding + Canny + 1D peak detection along the strip. If noisy, train a small UNet on ~200 manually labeled frames.
   - Output: time series `η_px(t)` of waterline pixel row

4. **Detrending + scaling**
   - Subtract a slow-moving median (low-pass at ~30 s) to remove tide
   - Convert `η_px(t) → η(t)` in meters using vertical scale from calibration
   - This gives surface elevation at one fixed location, sampled at video framerate

5. **Wave statistics** (`stats.py`)
   - Zero-crossing analysis on η(t) over rolling 17-min windows (matches CDIP standard)
   - Compute H_1/3 (= Hs), mean period Tz, peak period Tp from zero-crossing or via FFT spectrum
   - Output: timestamped DataFrame of Hs, Tp

6. **Validation** (`validate.py`)
   - Pull CDIP 201 spectra over the same window via erddapy
   - Compute RMSE, bias, scatter index, R² between cam-derived Hs and buoy Hs
   - Plot 1:1 scatter, time-series overlay, residuals vs swell direction/period

**Phase 1 success criterion:** RMSE on Hs ≤ 25 cm against the co-located buoy for sea states between 0.5–2 m. That's a defensible result — well within Surfline's published accuracy expectations for forecasts.

## Phase 2 — Multi-cam generalization (2–3 weeks)

Generalize to sites without a pier. Two scaling strategies:

**A. Stationary reference at known height** — lifeguard towers, jetty markers, navigation buoys. Requires site-specific annotation but transfers the waterline-tracking approach.

**B. Surfer-as-scale (your YOLOv8-pose wheelhouse)**
- Run pose on surfers in the lineup
- Use anatomical keypoint distances (e.g., shoulder-hip ~50 cm) as a moving scale ruler
- Measure wave face height in pixels relative to surfer; convert
- Less accurate (variable surfer size, perspective), so use as fallback or for unbroken faces only

Build a `Site` config schema so each new beach adds a YAML file with `cam_url`, `calibration_method`, `buoy_id`, and stationary-feature annotations. Adding a beach should be a config change, not a code change.

## Phase 3 — Validation + writeup (1 week)

- Run on 1–2 months of historical footage across 3–5 sites
- Failure mode analysis: night, fog, glassy conditions, big surf where the piling is in white water, PTZ events
- Compare cam-derived Hs against Surfline's published surf-height range for those days at those spots. Where do you disagree? Why? That's the most interesting finding for the application.

## Phase 4 — Wave quality (target: 2 weeks, after Phase 3 is solid)

Hs tells you *how big*. Quality tells you *how good*. Surfline already publishes human-rated quality ("Poor → Epic"); an automated, explainable version is more interesting to them than another height measurer. Build four components, expose them separately, then combine.

**4a. Peel angle** (`quality/peel.py`)
- Per-frame foam-line segmentation (white-pixel mask + morphological cleanup, or a small UNet)
- Track foam-line lateral propagation across consecutive frames
- Peel angle = angle between propagation vector and wave crest line
- Aggregate per wave: median peel angle, with 30–45° flagged as ideal, near-0° as closeout, >60° as mushy
- Background lit: search "peel angle remote sensing" / Holman & Stanley argus literature

**4b. Make rate** (`quality/make_rate.py`) — *the novel piece*
- Run YOLOv8-pose on lineup, persist tracks via ByteTrack or BoT-SORT
- Per surfer per attempted takeoff: ride duration from first standing pose to fall/kickout
- Define a make as ride duration > threshold (start with 4s, tune per spot)
- Spot-level metric: make rate = makes / attempted takeoffs, rolling 30-min window
- Self-supervised — no labels needed. Correlates directly with surfer-perceived quality.
- This is the angle nobody else publishes on. Lead with it in the writeup.

**4c. Surface texture classifier** (`quality/texture.py`)
- 4-class CNN: glassy / textured / chop / blown-out
- Train on ~500 hand-labeled 224×224 patches of water surface (away from breaking zones)
- Start with FFT-energy features as baseline before training a model — high-frequency band energy alone is often enough
- Output: dominant class + softmax probabilities per timestamp

**4d. Consistency metrics** (`quality/consistency.py`)
- Derive from the Hs and η(t) outputs already produced in Phase 1
- Set-vs-lull autocorrelation, wave-height variance, inter-set period
- Score: consistent overhead sets every 1–3 min ranks higher than scattered same-Hs-mean noise

**Composite Spot Quality Index**
- Weighted combination of peel-angle score × make-rate × texture-class × consistency
- Tune weights against Surfline's published quality ratings (scrape historical via Wayback for a held-out validation set — ToS-clean since you're only using ratings, not cam imagery)
- **Keep components visible.** A forecaster won't trust a single black-box score; they'll trust "make rate dropped to 22% as onshore wind kicked in at 9am." Build the dashboard accordingly.

**Phase 4 success criterion:** Composite SQI correlates ≥0.6 with Surfline's published quality ratings on a held-out month. Individual components are explainable to a non-ML forecaster.

## Repo structure

```
camwaveheight/
├── README.md           # Executive summary + result plots up top
├── pyproject.toml
├── configs/sites/
│   ├── scripps_pier.yaml
│   ├── blacks.yaml
│   └── trestles.yaml
├── src/camwaveheight/
│   ├── ingest.py
│   ├── calibration.py
│   ├── waterline.py
│   ├── pose_scale.py
│   ├── stats.py
│   ├── validate.py
│   ├── quality/
│   │   ├── peel.py
│   │   ├── make_rate.py
│   │   ├── texture.py
│   │   ├── consistency.py
│   │   └── sqi.py       # composite Spot Quality Index
│   └── pipeline.py     # orchestrates one site end-to-end
├── notebooks/
│   ├── 01_calibration_walkthrough.ipynb
│   └── 02_validation_results.ipynb
├── data/               # gitignored; .dvc or just READMEs pointing to sources
└── reports/
    └── validation_v1.pdf
```

## Tech stack

- Python 3.11, OpenCV, PyTorch (if UNet), NumPy/SciPy/Pandas
- `ultralytics` for YOLOv8-pose
- `segmentation-models-pytorch` if UNet waterline beats classical
- `erddapy` + `xarray` for CDIP buoy data
- `streamlit` for the demo dashboard

## Watch out for

- **PTZ events** — log and exclude frame ranges where the cam reframes. Calibration is per-view.
- **Refraction/shoaling** — Hs at the cam location ≠ Hs at the offshore buoy. Expect 10–30% amplification at the surf zone vs offshore. Validate against the *nearshore* CDIP station, not offshore.
- **Tide aliasing** — your detrending window must be much shorter than the M2 tidal period (12.4h) but long enough to preserve infragravity (>30s). 30s low-pass is a reasonable default.
- **Wave breaking** — η(t) is only well-defined where waves haven't broken. Pick a piling at the seaward end of the surf zone for cleanest signal.
- **Resolution/framerate** — most streamed cams are 30fps at 720p–1080p. Plenty for wave periods of 6–20s; not enough for crest velocity at high zoom-out.

## Deliverables for the Surfline application

1. Public GitHub repo with README that leads with the validation plot
2. 2-page PDF: approach, results, failure modes, what you'd do with their actual cam metadata + bathymetry
3. Streamlit demo running live on one site
4. Cover-letter hook: "RMSE 24 cm against CDIP 201 across 6 weeks at Scripps Pier — happy to walk you through the pipeline."

## Stretch (only if Phase 1–4 land cleanly)

- **Forecast validator** — pull historical Surfline-published Hs from Wayback or their public API where available, compute systematic bias by spot/swell direction. Most directly useful-to-them analysis.
- **Multi-cam fusion** at a single beach with overlapping views
- **Quality forecast** — train a model to predict SQI 12–48h out from swell/wind forecast inputs. Closes the loop from observation to prediction.
