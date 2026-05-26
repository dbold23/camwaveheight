# Lessons

## Calibration / wave detection
- **Always view a keyframe before designing the calibration approach.** The README assumed pier pilings would be visible in the Scripps cam; in reality the cam is mounted on the pier looking outward at open ocean — no pilings in frame. Pulled a single-frame snapshot from the HLS stream first; would have wasted hours building a homography module that had no input to calibrate against.
- **Surfline-style cams use regression calibration, not metric homography.** When a cam has no in-frame metric reference, fit pixel-space Hs against CDIP buoy Hs via linear regression on a training window. It's the only path that generalizes across a cam network. Document training window vs held-out window clearly.

## CDIP / ERDDAP
- **Don't trust placeholder IDs in spec docs.** README claimed Scripps Nearshore was CDIP 073; actual is 201. Always verify station IDs against the live ERDDAP catalog before wiring them into code or configs. Query: `https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.csv?station_id,metaStationName,latitude,longitude&distinct()`
- **CDIP ERDDAP uses one aggregated dataset, not per-station ones.** All wave parameters live in `wave_agg` with a `station_id` filter. There is no `<stn>p1_historic` table-level dataset.
- **erddapy's `to_pandas(parse_dates=...)` chokes on the ERDDAP units row.** Cleaner to call `requests.get(e.get_download_url(response="csv"))`, then `pd.read_csv(..., skiprows=[1])` to skip the units row, then parse times manually.
- **Always pass `waveFlagPrimary=1`** to filter to QC-passed observations. Without it you can get suspect or failed rows interleaved.
