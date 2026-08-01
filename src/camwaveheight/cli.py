"""Top-level CLI: `cwh <subcommand>`."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from camwaveheight import cdip, ingest
from camwaveheight.site import Site, WaveROI


@click.group()
@click.option("-v", "--verbose", is_flag=True)
def main(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@main.command("fetch-buoy")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True, help="ISO date, UTC, e.g. 2026-05-01")
@click.option("--end", required=True, help="ISO date, UTC")
@click.option("--cache", "cache_dir", type=click.Path(), default="data/cdip")
@click.option("--out", "out_path", type=click.Path(), default=None)
def fetch_buoy(site_path: str, start: str, end: str, cache_dir: str, out_path: str | None) -> None:
    """Pull CDIP parameter time series for a site's buoy and save to parquet."""
    site = Site.load(site_path)
    df = cdip.fetch_cdip_params(site.buoy_id, start, end, cache_dir=cache_dir)
    click.echo(f"{site.name}: {len(df)} rows, {df.index.min()}..{df.index.max()}")
    click.echo(df.describe().to_string())
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        click.echo(f"wrote {out_path}")


@main.command("buoy-plot")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--out", "out_path", type=click.Path(), default="reports/buoy_quicklook.png")
def buoy_plot(site_path: str, start: str, end: str, out_path: str) -> None:
    """Sanity quicklook: Hs/Tp time series for a site's buoy."""
    import matplotlib.pyplot as plt

    site = Site.load(site_path)
    df = cdip.fetch_cdip_params(site.buoy_id, start, end, cache_dir="data/cdip")
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    df["waveHs"].plot(ax=axes[0], color="C0")
    axes[0].set_ylabel("Hs (m)")
    axes[0].set_title(f"CDIP {site.buoy_id} — {site.name}: {start} → {end}")
    if "waveTp" in df.columns:
        df["waveTp"].plot(ax=axes[1], color="C1")
        axes[1].set_ylabel("Tp (s)")
    axes[1].set_xlabel("UTC")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    click.echo(f"wrote {out_path}  ({len(df)} samples)")


@main.command("set-roi")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--x", type=int, required=True)
@click.option("--y", type=int, required=True)
@click.option("--w", type=int, required=True)
@click.option("--h", type=int, required=True)
def set_roi(site_path: str, x: int, y: int, w: int, h: int) -> None:
    """Write the surf-zone ROI into the site YAML."""
    site = Site.load(site_path)
    site.calibration.wave_roi = WaveROI(x=x, y=y, w=w, h=h)
    site.dump(site_path)
    click.echo(f"{site.name}: ROI set to x={x} y={y} w={w} h={h}")


@main.command("run")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--out-root", type=click.Path(), default="data/raw")
@click.option("--cache-eta", type=click.Path(), default="data/eta/eta_px.parquet")
@click.option("--cdip-cache", type=click.Path(), default="data/cdip")
@click.option("--train-frac", type=float, default=0.7)
@click.option("--window-min", type=float, default=17.0, help="Rolling-Hs window in minutes.")
@click.option("--step-min", type=float, default=5.0, help="Rolling-Hs step in minutes.")
@click.option("--tag", default="v1")
def run_pipeline(
    site_path: str,
    out_root: str,
    cache_eta: str,
    cdip_cache: str,
    train_frac: float,
    window_min: float,
    step_min: float,
    tag: str,
) -> None:
    """End-to-end: extract η_px → rolling Hs → fit to CDIP 201 → plots."""
    from camwaveheight.pipeline import run_pipeline as _run

    _run(
        site_path,
        out_root=out_root,
        cache_eta=cache_eta,
        cdip_cache=cdip_cache,
        train_frac=train_frac,
        tag=tag,
        window_sec=int(window_min * 60),
        step_sec=int(step_min * 60),
    )


@main.command("record")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--hours", type=float, default=None, help="Stop after N hours. Omit to run until killed.")
@click.option("--segment-min", type=int, default=10, help="Segment length in minutes.")
@click.option("--out-root", type=click.Path(), default="data/raw")
@click.option("--resilient/--single", default=True,
              help="Resilient mode auto-restarts ffmpeg on death (for multi-day capture).")
def record_cmd(
    site_path: str, hours: float | None, segment_min: int, out_root: str, resilient: bool
) -> None:
    """Record a site's HLS stream into segmented MP4s with UTC-timestamped filenames."""
    site = Site.load(site_path)
    if not site.cam_url:
        raise click.ClickException(f"{site.name}: cam_url is not set in {site_path}")
    duration = int(hours * 3600) if hours else None
    mode = "resilient (auto-restart)" if resilient else "single-shot"
    click.echo(f"recording {site.name} from {site.cam_url}  [{mode}]")
    click.echo(f"  out_root={out_root}  segment={segment_min}min  duration={hours}h")
    if resilient:
        ingest.record_resilient(
            url=site.cam_url,
            site_name=site.name,
            out_root=out_root,
            segment_sec=segment_min * 60,
            referer=site.cam_referer,
            total_duration_sec=duration,
        )
        click.echo("recording supervisor stopped")
    else:
        rc = ingest.record(
            url=site.cam_url,
            site_name=site.name,
            out_root=out_root,
            segment_sec=segment_min * 60,
            referer=site.cam_referer,
            duration_sec=duration,
        )
        click.echo(f"ffmpeg exited rc={rc}")


# --------------------------------------------------------------------------- #
# Satellite data stream (sibling to the cam pipeline). All heavy deps are
# imported lazily inside each command; missing deps/creds surface as a clean
# ClickException with the exact install / login step.
# --------------------------------------------------------------------------- #


def _sat_guard(fn, *args, **kwargs):
    """Run a satellite fetch, mapping expected setup failures to ClickException."""
    from camwaveheight.satellite.creds import CredentialsError

    try:
        return fn(*args, **kwargs)
    except (ModuleNotFoundError, CredentialsError, NotImplementedError) as exc:
        raise click.ClickException(str(exc))


@main.command("sat-waves")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True, help="ISO date, UTC")
@click.option("--end", required=True, help="ISO date, UTC")
@click.option(
    "--product",
    type=click.Choice(["model", "reanalysis", "altimeter", "all"]),
    default="model",
)
@click.option("--out", "out_path", type=click.Path(), default=None)
def sat_waves(site_path: str, start: str, end: str, product: str, out_path: str | None) -> None:
    """Fetch CMEMS satellite/model wave products (model / reanalysis / altimeter)."""
    from camwaveheight.satellite import waves

    site = Site.load(site_path)
    jobs = {
        "model": waves.fetch_cmems_wave_model,
        "reanalysis": waves.fetch_cmems_wave_reanalysis,
        "altimeter": waves.fetch_altimeter_swh,
    }
    wanted = list(jobs) if product == "all" else [product]
    results = {name: _sat_guard(jobs[name], site, start, end) for name in wanted}

    for name, df in results.items():
        if len(df):
            click.echo(f"[{name}] {len(df)} rows, {df.index.min()}..{df.index.max()}")
            click.echo(df.describe().to_string())
        else:
            click.echo(f"[{name}] 0 rows (expected for sparse altimeter / narrow windows)")
    if out_path:
        name, df = next(iter(results.items()))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        click.echo(f"wrote {out_path} ({name})")


@main.command("sat-validate")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True, help="Buoy/model window start, UTC")
@click.option("--end", required=True, help="Buoy/model window end, UTC")
@click.option("--cache-eta", type=click.Path(), default="data/eta/eta_px.parquet")
@click.option("--cdip-cache", type=click.Path(), default="data/cdip")
@click.option("--train-frac", type=float, default=0.7)
@click.option("--altimeter/--no-altimeter", default=False, help="Also fetch + compare altimeter SWH.")
@click.option("--tag", default="sat_v1")
def sat_validate(
    site_path: str,
    start: str,
    end: str,
    cache_eta: str,
    cdip_cache: str,
    train_frac: float,
    altimeter: bool,
    tag: str,
) -> None:
    """Compare cam Hs vs CDIP buoy vs CMEMS model (+ optional altimeter)."""
    from camwaveheight import stats, validate
    from camwaveheight.satellite import waves

    site = Site.load(site_path)
    if not Path(cache_eta).exists():
        raise click.ClickException(
            f"no cam eta cache at {cache_eta}; run `cwh run --site {site_path}` first."
        )
    eta = pd.read_parquet(cache_eta)
    hs_df = stats.rolling_hs(eta["eta_px"])
    if hs_df.empty:
        raise click.ClickException("rolling_hs produced no windows from the cached eta.")

    buoy = cdip.fetch_cdip_params(site.buoy_id, start, end, cache_dir=cdip_cache)
    if buoy.empty:
        raise click.ClickException(f"no CDIP {site.buoy_id} rows in [{start}, {end}].")
    paired = validate.align_to_buoy(hs_df, buoy, cam_col="hs_px_4std")
    if len(paired) < 6:
        raise click.ClickException(f"only {len(paired)} cam/buoy pairs; need ≥6 to fit.")
    fit, paired = validate.fit_train_test(paired, train_frac=train_frac)

    model = _sat_guard(waves.fetch_cmems_wave_model, site, start, end)
    paired = validate.merge_source(paired, model, "model_hs", tolerance="90min")
    sources = ["hs_pred_m", "model_hs"]
    if altimeter:
        alt = _sat_guard(waves.fetch_altimeter_swh, site, start, end)
        if not alt.empty:
            paired = validate.merge_source(paired, alt, "alt_swh", tolerance="90min")
            sources.append("alt_swh")
        else:
            click.echo("altimeter: 0 rows in window — omitted from comparison")

    table = validate.compare_sources(paired, sources=tuple(sources))
    click.echo(f"\n{site.name}: source comparison vs CDIP {site.buoy_id} (reference=waveHs)")
    click.echo(table.to_string())
    plots = validate.plot_three_way(paired, sources=tuple(sources), tag=tag)
    out_json = Path("reports") / f"sat_validate_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(table.reset_index().to_json(orient="records", indent=2))
    click.echo(f"wrote {plots['timeseries']}, {plots['scatter']}, {out_json}")


@main.command("sat-env")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--vars", "vars_csv", default="sst", help="Comma list ⊆ sst,wind,chl")
@click.option("--out", "out_path", type=click.Path(), default=None)
def sat_env(site_path: str, start: str, end: str, vars_csv: str, out_path: str | None) -> None:
    """Build a daily SST/wind/ocean-color panel (SST via no-auth NOAA CRW)."""
    from camwaveheight.satellite import environment

    site = Site.load(site_path)
    variables = tuple(v.strip() for v in vars_csv.split(",") if v.strip())
    panel = _sat_guard(environment.build_env_panel, site, start, end, variables=variables)
    if panel.empty:
        click.echo("env panel empty — no source returned data for the window.")
        return
    click.echo(f"env panel: {len(panel)} days, cols={list(panel.columns)}")
    click.echo(panel.describe().to_string())
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out_path)
        click.echo(f"wrote {out_path}")


@main.command("sat-shoreline")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--out", "out_path", type=click.Path(), default=None)
def sat_shoreline(site_path: str, start: str, end: str, out_path: str | None) -> None:
    """Extract satellite-derived shoreline positions via CoastSat (sat-shoreline extra)."""
    from camwaveheight.satellite import shoreline

    site = Site.load(site_path)
    df = _sat_guard(shoreline.extract_shoreline_timeseries, site, start, end, out_dir=out_path)
    click.echo(f"{site.name}: {len(df)} shoreline observations")
    if len(df):
        click.echo(df.head().to_string())


@main.command("sat-quicklook")
@click.option("--site", "site_path", type=click.Path(exists=True), required=True)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--out", "out_path", type=click.Path(), default="reports/sat_quicklook.png")
def sat_quicklook(site_path: str, start: str, end: str, out_path: str) -> None:
    """Quicklook: CMEMS model Hs overlaid on the site's CDIP buoy Hs."""
    import matplotlib.pyplot as plt

    from camwaveheight.satellite import waves

    site = Site.load(site_path)
    buoy = cdip.fetch_cdip_params(site.buoy_id, start, end, cache_dir="data/cdip")
    model = _sat_guard(waves.fetch_cmems_wave_model, site, start, end)
    fig, ax = plt.subplots(figsize=(10, 4))
    if not buoy.empty:
        ax.plot(buoy.index, buoy["waveHs"], "o-", color="C0", ms=3, label=f"CDIP {site.buoy_id}")
    if not model.empty:
        ax.plot(model.index, model["model_hs"], ".-", color="C1", label="CMEMS model")
    ax.set_ylabel("Hs (m)")
    ax.set_xlabel("UTC")
    ax.set_title(f"{site.name}: model vs buoy  {start} → {end}")
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    click.echo(f"wrote {out_path}  (buoy={len(buoy)}, model={len(model)})")


if __name__ == "__main__":
    main()
