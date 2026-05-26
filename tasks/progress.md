# Progress

## Where we stand — 2026-05-25 evening

### Goal
`/goal Phase 1 MVP complete — RMSE on cam-derived Hs ≤ 25 cm vs CDIP 201 over a 24h window at Scripps Pier`

### Done
- Repo scaffolded; public on GitHub: https://github.com/dbold23/camwaveheight
- CDIP buoy fetcher (`cdip.py`) — live-tested, parquet-cached
- HLS recorder (`ingest.py`) — ffmpeg-based, segmented MP4s with UTC-timestamp filenames
- Pivot to **regression calibration** (away from pier-piling homography — the cam doesn't see pilings)
- `wave_detect.py`: per-frame topmost vertically-coherent foam row in surf-zone ROI
  - 96.6% detection rate on a 60s evening test clip
  - η_px std ≈ 24 px; clear wave-period oscillation visible
- `stats.py`: rolling Hs (4·std + zero-crossing) + Tp via Welch
- `validate.py`: align cam Hs to CDIP, train/test split, linear regression, RMSE/bias/R²/SI plots
- `pipeline.py`: end-to-end orchestrator; cam-only diagnostic when no buoy overlap
- Smoke-test on first 10-min recorded segment passed: Tp 11-17s, consistent with this week's swell at CDIP 201

### Running in background
- **25-hour HLS recording of `wc-scripps`** — started 2026-05-26T02:47Z, ends ~2026-05-27T03:47Z
- ~17 MB/min ≈ 25 GB total on the external drive
- Background process id: tracked by harness; check segments under `data/raw/scripps_pier/`

### Blocker — wait for time alignment
CDIP ERDDAP currently has data through **2026-05-25T14:30Z**; cam recording started **2026-05-26T02:47Z**. CDIP lags realtime by ~6-12 h.

When to re-run the pipeline:
- **+6 h**: first ~1-2 cam/buoy paired samples may be available
- **+12 h**: ~6-12 paired samples; first meaningful regression
- **+24-30 h** (sometime 2026-05-26 evening or 2026-05-27): a clean 13+ hour overlap window, enough to run the full pipeline with the default 17-min CDIP-standard window

### How to re-run
```bash
cd "<repo root>"
.venv/bin/cwh run --site configs/sites/scripps_pier.yaml --tag v1
```
Or for a partial-data smoke-test with a smaller window:
```bash
.venv/bin/cwh run --site configs/sites/scripps_pier.yaml --tag smoke --window-min 5 --step-min 1
```

### Open decisions / next iterations
- Foam thresholds were tuned on a dim evening clip — may need adjustment for daytime footage (likely tighter, since bright daytime water gives more false positives)
- ROI may need refinement after seeing daytime footage — currently x=100 y=420 w=1080 h=280
- If RMSE doesn't hit 25 cm, options: enrich the η_px signal with foam_count, fit per-period subsets (different coefficient for long-period vs short-period swell), or try a quadratic instead of linear regression

### Known issues / minor TODOs
- `extract_eta_for_site` logs a warning for the actively-being-written segment (moov atom not found) — harmless but noisy
- No safeguard against the background recording being killed; if it dies, restart with `cwh record --site configs/sites/scripps_pier.yaml --hours 25`
