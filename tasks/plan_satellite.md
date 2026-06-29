# Plan A — Satellite data stream (waves + shoreline + environment)

## Context

`camwaveheight` today is a single-signal pipeline: ffmpeg HLS recorder → per-frame **motion energy**
`mean(|I_t − I_{t-1}|)` in one fixed surf-zone ROI → rolling 17-min Hs/Tp → linear regression vs CDIP buoy 201.
It met Phase 1 (RMSE 7 cm random / 15.8 cm sequential vs the buoy) but produces only two scalar parquet tables
(`motion.parquet`, `eta_px.parquet`) over ~26 h of footage at Scripps Pier, La Jolla.

This plan adds a new `src/camwaveheight/satellite/` subpackage, sibling to the cam pipeline, that pulls
satellite + model ocean products for the site location/time windows, caches them like `cdip.py`, and integrates
them as (a) extra wave validation/forecast ground truth, (b) shoreline/nearshore change, (c) environmental context.

Satellite is additive: the validated motion-energy Hs pipeline stays the system of record; satellite is a sibling
data stream. The existing `cdip.py` fetch/cache pattern, `validate.align_to_buoy`/`fit_train_test`, `site.Site`
schema, and `ingest.segment_start_utc` timing are reused throughout.

### Conventions to mirror (verified in repo)
- `cdip.py`: module constants, `_iso()` UTC coercion, `fetch_*(... cache_dir=...)`, deterministic cache key
  `f"...{start}_{end}.parquet".replace(":", "")`, "read cache if exists else fetch", returns UTC-indexed sorted
  `DataFrame`, empty DataFrame on no-match. **Heavy deps imported lazily inside functions** (new deps are optional extras).
- `validate.align_to_buoy` normalizes both indices to UTC `ns` then `merge_asof`. `data/` and `*.nc` already gitignored.

### Package layout
```
src/camwaveheight/satellite/
  __init__.py        # light, lazy
  common.py          # iso_utc, cache_key, bbox_from_site, coerce_utc_index, haversine_km, load_from_cache_or
  creds.py           # CMEMS / GEE / Earthdata credential resolution (env → .netrc → provider config)
  waves.py           # CMEMS WAVERYS + analysis-forecast + altimeter L3 SWH  (cdip.py twin)
  shoreline.py       # CoastSat/GEE satellite-derived shoreline + transects
  environment.py     # SST, wind, chlorophyll/turbidity
configs/sites/aoi/scripps_pier_aoi.geojson   # small committable AOI polygon (~1.5km along × 0.3km cross-shore)
```

### Site schema changes (`site.py`) — all optional/defaulted so existing YAML still validates
- `GeoLocation{lat, lon}`; `AOI{bbox:(lonmin,latmin,lonmax,latmax)|None, polygon_geojson:str|None, subset_radius_deg=0.25}`
- `SatelliteProducts{ cmems_wave_model_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
  cmems_wave_reanalysis_id="cmems_mod_glo_wav_my_0.2deg_PT3H-i",
  cmems_altimeter_l3_id="cmems_obs-wave_glo_phy-swh_nrt_multi-l3_PT1S", altimeter_max_dist_km=50.0,
  sst_dataset="MUR", wind_dataset="ASCAT", ocean_color_dataset="S3_OLCI", sds_collections=("S2","L8","L9"),
  sds_max_cloud_pct=20.0 }`
- `SatelliteConfig{enabled=False, products=SatelliteProducts(), cache_root="data/satellite"}`
- Add to `Site`: `location|None`, `aoi|None`, `satellite=SatelliteConfig()`.
- Append `location/aoi/satellite` blocks to `configs/sites/scripps_pier.yaml` (lat 32.867, lon -117.257,
  bbox `[-117.30,32.82,-117.21,32.91]`).

### Dependencies (`pyproject.toml`, new optional extras)
- `sat-waves = ["copernicusmarine>=1.3"]` (MEDIUM: pulls boto3/zarr/dask)
- `sat-shoreline = ["earthengine-api>=0.1.390","geemap>=0.32","geopandas>=0.14","shapely>=2.0","scikit-image>=0.22"]`
  (HEAVY; CoastSat itself via PyPI `coastsat` or `pip install git+https://github.com/kvos/CoastSat`)
- `sat-env = ["pystac-client>=0.7","odc-stac>=0.3","rioxarray>=0.15","requests>=2.31"]`
- `sat-all = camwaveheight[sat-waves,sat-shoreline,sat-env]`
- Note: GDAL-backed wheels (geopandas/rioxarray) are finicky on macOS — install shoreline extras only when on P3, consider conda-forge.

### Accounts / secrets (`creds.py`; never commit)
| Provider | Used by | Auth |
|---|---|---|
| Copernicus Marine | waves (model/reanalysis/altimeter), opt. wind/SST | `copernicusmarine login` or env `COPERNICUSMARINE_SERVICE_USERNAME/PASSWORD` |
| Google Earth Engine | shoreline (CoastSat) | `earthengine authenticate`; project via env `EE_PROJECT` |
| NASA Earthdata / PODAAC | MUR SST, ASCAT wind, ocean color | `~/.netrc` machine `urs.earthdata.nasa.gov` |
| NOAA CoastWatch/NOMADS/ERDDAP | WW3, CRW SST fallbacks | none |
- `.gitignore` additions: `.env`, `*.netrc`, `.copernicusmarine/`, `configs/secrets*.yaml`.

### Modules
**`waves.py`** (mirror cdip.py):
- `fetch_cmems_wave_model(site,start,end,product_id=None,cache_dir=...)`: lazy `copernicusmarine.subset` over bbox+time
  (vars `VHM0`/Hs, `VTPK`/Tp, `VMDR`/dir), open xarray, `.sel(lat,lon,method="nearest")` → cols `model_hs,model_tp,model_dp`,
  UTC index; cache parquet (+ raw `.nc`). Global anfc ~0.083° (~8km) 3-hourly NRT; WAVERYS reanalysis 0.2° (~22km) multi-year, ~months latency.
- `fetch_altimeter_swh(...)`: CMEMS WAVE_GLO_PHY_SWH L3 multi-mission along-track; haversine filter ≤ max_dist_km of site;
  cols `alt_swh,alt_dist_km,alt_mission`. **Feasibility:** sparse revisit, land-contaminated <10–20 km from coast → expect empty windows; cross-check only.
- `latest_model_hs(site,hours=48)`. NOAA WW3 no-auth fallback documented (P3).

**`validate.py` extensions** (additive, don't break 2-way):
- `align_three_way(cam_hs,buoy,model_or_sat,ref_col="model_hs",tolerance="90min")` — start from `align_to_buoy`, merge_asof model onto buoy timeline (model 3-hourly → larger tol).
- `compare_sources(paired3, sources=("hs_pred_m","model_hs","alt_swh"), reference="waveHs")` — RMSE/bias/R²/SI per source; factor metric math into shared `_metrics(pred,ref)` reused by `fit_train_test`.
- `plot_three_way(...)` styled like `plot_validation`.

**`shoreline.py`**: primary toolchain **CoastSat (GEE)** for S2/L8/L9 satellite-derived shoreline (cloud mask, sub-pixel, tidal correction).
- `extract_shoreline_timeseries(site,start,end)` → shoreline vectors + cross-shore position time series along transects → `data/satellite/shoreline/shoreline_position.parquet` (UTC index, per-transect chainage m).
- `define_transects(site)`, `tidal_correction(...)`, `fetch_sdb(...)` (P3 stub; SDB limited by surf-zone turbidity).
- **Cross-reference contract with Plan B:** share transect IDs + a `cross_shore_position_m` column on a UTC index so the cam's own shoreline segmentation can later be regressed against absolute SDS (same way cam Hs → CDIP).
- **Feasibility:** 5-day revisit, cloud-limited, ~5–10 m accuracy — slow-change monitor, not real-time.

**`environment.py`** (small UTC parquets under `data/satellite/env/`):
- SST: primary MUR (PODAAC ~1km daily, Earthdata) ; easy fallback **NOAA Coral Reef Watch 5km daily via CoastWatch ERDDAP** (reuse cdip.py erddapy flow) — recommend CRW-ERDDAP for P2 simplicity. `fetch_sst(...)→ sst_c`.
- Wind: ASCAT (PODAAC ~12.5–25km) or ERA5/CMEMS fallback. `fetch_wind(...)→ wind_speed,wind_dir`.
- Ocean color: Sentinel-3 OLCI (300m) via STAC, or NASA OC L3 (4km). `fetch_ocean_color(...)→ chl_mg_m3,turbidity`.
- `build_env_panel(site,start,end)` outer-joins onto daily UTC index → `env_panel.parquet`.

### CLI (`cli.py`, lazy imports + `raise click.ClickException` with precise "install extra X / run login" messages)
- `cwh sat-waves --site --start --end [--product model|reanalysis|altimeter|all] [--out]`
- `cwh sat-validate --site --start --end [--tag]` (cam vs CDIP vs CMEMS model → metric table + plots/JSON)
- `cwh sat-shoreline --site --start --end [--out]`
- `cwh sat-env --site --start --end [--vars sst,wind,chl] [--out]`
- `cwh sat-quicklook` (parallel to `buoy-plot`)

### Caching
```
data/satellite/waves/cmemswave_<id>_<start>_<end>.parquet (+ .nc)  altimeter_<id>_..._.parquet
data/satellite/shoreline/shoreline_position.parquet, shorelines.geojson, transects.geojson
data/satellite/env/sst_<src>_..._.parquet, wind_..., chl_..., env_panel.parquet
```

### Phases + verification
- **P0 — wave vertical slice:** schema fields + scripps yaml, `common.py`, `creds.py`(CMEMS), `waves.fetch_cmems_wave_model`,
  `cwh sat-waves`, `sat-waves` extra, gitignore. Verify: `pip install -e ".[sat-waves]"`; `copernicusmarine login`;
  `cwh sat-waves --product model --start 2026-05-07 --end 2026-05-28` → non-empty parquet, `model_hs` in 0.3–2.5 m;
  `cwh sat-validate` → model-vs-CDIP RMSE small (<0.3 m) proves alignment/units.
- **P1 — waves complete + 3-way:** altimeter, reanalysis, `align_three_way`/`compare_sources`/`plot_three_way`, `_metrics`.
  Verify: wide-window altimeter returns ≥1 row; `sat-validate` prints 3-source table vs CDIP.
- **P2 — env panel:** `environment.py` (CRW-ERDDAP SST first, then ASCAT wind, then chl), `sat-env`. Verify:
  `cwh sat-env --vars sst,wind` → `sst_c` ~16–20°C, `wind_speed` populated, daily index.
- **P3 — shoreline (heaviest, last):** AOI geojson + transects, CoastSat `shoreline.py`, `sat-shoreline`, GEE auth, SDB/WW3 stubs.
  Verify: `cwh sat-shoreline --start 2026-01-01 --end 2026-06-01` → `shoreline_position.parquet` ≥3 cloud-free dates; geojson overlays the pier in QGIS.

### Tests (mock network): `test_common.py` (cache key/bbox/haversine/UTC), `test_validate_threeway.py` (synthetic→known metrics), `test_waves_cache.py` (monkeypatch subset→tiny .nc→parquet). Credentialed/network behind `@pytest.mark.network` skipped by default.

### Critical files: `cdip.py` (mirror), `validate.py` (extend), `site.py` (add fields), `cli.py` (add subcommands), `pyproject.toml` (extras).
