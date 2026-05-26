"""Top-level CLI: `cwh <subcommand>`."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from camwaveheight import cdip
from camwaveheight.site import Site


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


if __name__ == "__main__":
    main()
