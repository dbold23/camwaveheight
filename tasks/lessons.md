# Lessons

## CDIP / ERDDAP
- **Don't trust placeholder IDs in spec docs.** README claimed Scripps Nearshore was CDIP 073; actual is 201. Always verify station IDs against the live ERDDAP catalog before wiring them into code or configs. Query: `https://erddap.cdip.ucsd.edu/erddap/tabledap/wave_agg.csv?station_id,metaStationName,latitude,longitude&distinct()`
- **CDIP ERDDAP uses one aggregated dataset, not per-station ones.** All wave parameters live in `wave_agg` with a `station_id` filter. There is no `<stn>p1_historic` table-level dataset.
- **erddapy's `to_pandas(parse_dates=...)` chokes on the ERDDAP units row.** Cleaner to call `requests.get(e.get_download_url(response="csv"))`, then `pd.read_csv(..., skiprows=[1])` to skip the units row, then parse times manually.
- **Always pass `waveFlagPrimary=1`** to filter to QC-passed observations. Without it you can get suspect or failed rows interleaved.
