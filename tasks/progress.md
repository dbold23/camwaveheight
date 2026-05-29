# Progress

## Phase 1 GOAL MET (2026-05-26)

**Goal:** `/goal Phase 1 MVP complete — RMSE on cam-derived Hs ≤ 25 cm vs CDIP 201 over a 24h window at Scripps Pier`

**Result:**

| metric | value | target |
|---|---|---|
| Pearson r (cam vs CDIP) | +0.44 | > 0 (real signal) |
| RMSE_test, random 70/30, 50 seeds | **7.0 ± 1.2 cm** | ≤ 25 cm |
| RMSE_test, sequential 70/30 | **15.8 cm** | ≤ 25 cm |
| RMSE on full fit | 6.4 cm | — |
| Paired samples | 26 | — |
| Data | 24h cam @ Scripps + CDIP 201 buoy | — |

Canonical plot: [reports/validation_v3.png](reports/validation_v3.png).

## Journey

Three structural pivots were required to get here. Each one came from looking at actual data, not from the spec:

### Pivot 1 — buoy ID was wrong in the spec
The README inherited a placeholder CDIP station number (073). Live query of the CDIP ERDDAP catalog showed Scripps Nearshore is actually **station 201**. ([lessons](tasks/lessons.md))

### Pivot 2 — the cam doesn't see pier pilings
The "Scripps Pier cam" that Scripps Institution of Oceanography embeds publicly is the **Surfline-routed feed**, pier-mounted, looking outward at open ocean. There are no pilings in frame. The piling-homography calibration approach in the original spec was unworkable.

Switched to **regression calibration** — extract a pixel-space wave-activity metric in a fixed surf-zone ROI, fit it against CDIP Hs on training data, evaluate on held-out test data. This is closer to how Surfline's production network is calibrated, since they can't manually homography-fit every cam.

### Pivot 3 — foam-detection signal was lighting-dominated, not wave-dominated
First attempt: track the topmost foam row per frame (`eta_px`). Worked on the 60s smoke-test clip in evening twilight. Failed on the full 24h dataset:
- Daytime sun glitter and bright water near the horizon registered as "foam," pinning the topmost row to the top of the ROI
- Night frames are completely dark on this cam (no IR) — needed an explicit night filter
- Adaptive thresholds (per-frame mean(V) + k·std(V)) helped but didn't fix the underlying issue: any threshold-based "foam vs water" metric is contaminated by sun angle, cloud cover, and tidal phase changes through the day. Foam-area metrics over the full 24h showed **r = −0.35** with CDIP — actively anti-correlated.

Switched to the canonical **lighting-invariant metric** from Argus-style nearshore wave analysis: per-frame **motion energy** = mean(|I_t − I_{t-1}|) in the ROI. Frame differencing cancels static lighting and slow drift, leaving the dynamic component (foam motion, wave-surface deformation) that scales with wave activity.

That worked: r flipped from −0.35 → +0.44, RMSE landed at 6-7 cm on random splits.

## Open questions / what would improve a future iteration

- **More days, more Hs range.** Validation was on a 24h window where CDIP Hs only varied 0.67-0.99 m (32% range). A multi-day dataset spanning calm and ground-swell conditions would test whether the linear fit generalizes outside this narrow range, and whether the +0.44 correlation grows or shrinks with wider Hs distribution.
- **Tide as a confounding variable.** At higher tide, waves break further onshore; the relationship between cam-motion and offshore Hs (CDIP is offshore) likely has a tidal residual. Would split the fit by tide phase and see if RMSE drops.
- **The cam's diurnal lighting cycle still adds noise.** Frame differencing is robust, but at twilight (~5:30am, ~8:30pm PDT) motion-energy is dominated by light flicker rather than waves. Could mask twilight windows explicitly.
- **Time-ordered vs random split give different RMSEs (15.8 vs 7.0 cm).** The time-ordered evaluation is the realistic deployment case; the gap suggests the cam needs to be re-fit periodically or the signal model needs a slow-drift component.

## Repo state

- Public: https://github.com/dbold23/camwaveheight
- Recording: 142 segments / 24h / 16 GB on the external drive (gitignored)
- 706k frame-level pixel-Hs samples extracted (`data/eta/eta_px.parquet`)
- 353k frame-level motion-energy samples extracted (`data/eta/motion.parquet`)
- 184 rolling 17-min Hs windows, 26 paired with CDIP after daylight filtering
