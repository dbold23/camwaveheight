"""Top-level CLI: `cwh <subcommand>`."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

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


if __name__ == "__main__":
    main()
