"""HLS stream recorder.

Spawns ffmpeg with segmenting so we get one MP4 per fixed interval, each
named with the UTC start time of its first packet. Designed to run
unattended for tens of hours; survives transient network failures via
ffmpeg's `-reconnect*` flags.

Usage (CLI):
    cwh record --site configs/sites/scripps_pier.yaml --hours 24

Output layout:
    <out_root>/<site_name>/<YYYYmmdd>/seg-YYYYmmddTHHMMSSZ.mp4
    <out_root>/<site_name>/manifest.csv   # appended per segment, UTC start + path

The manifest is the ground-truth time index — ffmpeg's strftime filename
captures wall-clock start of each segment, so frame timestamps inside the
segment are filename_start + frame_index / fps.
"""

from __future__ import annotations

import logging
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_SEGMENT_SEC = 600  # 10-min segments — small enough for safe restart, big enough to be cheap
DEFAULT_OUT_ROOT = Path("data/raw")


def _ffmpeg_cmd(
    url: str,
    out_pattern: str,
    segment_sec: int,
    referer: str | None,
    log_path: Path,
) -> list[str]:
    """Build the ffmpeg command. Re-encodes to nothing (stream copy) for cheapness.

    `out_pattern` must contain ffmpeg strftime tokens (e.g. %Y%m%dT%H%M%SZ).
    """
    cmd: list[str] = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    if referer:
        # ffmpeg requires a trailing \r\n on the headers blob.
        cmd += ["-headers", f"Referer: {referer}\r\n"]
    cmd += [
        # Robustness against transient TCP/TLS errors only. Do NOT set
        # -reconnect_at_eof or -reconnect_streamed here: HLS playlists
        # legitimately return EOF between updates, and those flags trigger
        # reconnect storms that CloudFront rejects.
        "-reconnect", "1",
        "-reconnect_delay_max", "10",
        "-rw_timeout", "30000000",  # 30s, microseconds
        "-i", url,
        # Stream copy — no transcode, preserves source fps/codec/quality.
        "-c", "copy",
        # Segment muxer with wall-clock-named outputs (UTC).
        "-f", "segment",
        "-segment_time", str(segment_sec),
        "-reset_timestamps", "1",
        "-strftime", "1",
        "-segment_format", "mp4",
        "-movflags", "+faststart",
        out_pattern,
    ]
    log.info("ffmpeg cmd: %s", " ".join(shlex.quote(c) for c in cmd))
    log.info("ffmpeg stderr -> %s", log_path)
    return cmd


def record(
    url: str,
    site_name: str,
    out_root: str | Path = DEFAULT_OUT_ROOT,
    segment_sec: int = DEFAULT_SEGMENT_SEC,
    referer: str | None = None,
    duration_sec: int | None = None,
) -> int:
    """Record an HLS stream into segmented MP4s.

    Args:
        url: HLS playlist URL.
        site_name: subfolder under out_root.
        out_root: root directory for recordings.
        segment_sec: target segment duration.
        referer: optional Referer header (some CDNs require it).
        duration_sec: stop after N seconds; None = run until killed.

    Returns:
        ffmpeg process exit code.
    """
    now = datetime.now(timezone.utc)
    day_dir = Path(out_root) / site_name / now.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(out_root) / site_name / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"ffmpeg-{now.strftime('%Y%m%dT%H%M%SZ')}.log"

    # ffmpeg strftime tokens — UTC because we passed TZ=UTC in env below.
    out_pattern = str(day_dir / "seg-%Y%m%dT%H%M%SZ.mp4")
    cmd = _ffmpeg_cmd(url, out_pattern, segment_sec, referer, log_path)

    env = {"TZ": "UTC"}
    import os
    env = {**os.environ, **env}

    with open(log_path, "ab") as logf:
        try:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, env=env)
        except FileNotFoundError:
            log.error("ffmpeg not found on PATH")
            return 127

        if duration_sec is None:
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                rc = proc.wait()
        else:
            try:
                rc = proc.wait(timeout=duration_sec)
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:
                    rc = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()

    log.info("ffmpeg exited rc=%d", rc)
    return rc


def list_segments(site_name: str, out_root: str | Path = DEFAULT_OUT_ROOT) -> list[Path]:
    """All recorded segments for a site, sorted by UTC start."""
    root = Path(out_root) / site_name
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("seg-*.mp4") if p.is_file())


def segment_start_utc(path: Path) -> datetime:
    """Parse the UTC start time from a segment filename like seg-20260525T193400Z.mp4."""
    stem = path.stem  # "seg-20260525T193400Z"
    ts = stem.split("-", 1)[1].rstrip("Z")
    return datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    sys.exit(record(url=sys.argv[1], site_name=sys.argv[2]))
