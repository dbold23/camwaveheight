"""Credential resolution for satellite providers.

Secrets never live in the repo. Each resolver checks env vars first, then the
provider's on-disk login cache / ~/.netrc, and raises a precise, actionable
`CredentialsError` (naming the exact login command or env var) when nothing is
found. Callers surface these as `click.ClickException` at the CLI boundary.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


class CredentialsError(RuntimeError):
    """Raised when a provider's credentials cannot be resolved."""


def cmems_credentials() -> tuple[str | None, str | None]:
    """Copernicus Marine credentials for `copernicusmarine` (waves, opt. SST/wind).

    Resolution order:
      1. env `COPERNICUSMARINE_SERVICE_USERNAME` / `COPERNICUSMARINE_SERVICE_PASSWORD`
      2. the on-disk credentials file written by `copernicusmarine login`

    Returns `(user, pwd)` when taken from the env, or `(None, None)` when relying
    on the on-disk login cache (the SDK reads it itself). Raises otherwise.
    """
    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME")
    pwd = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")
    if user and pwd:
        return user, pwd

    cache_dir = os.environ.get("COPERNICUSMARINE_CACHE_DIRECTORY")
    candidates = [Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials"]
    if cache_dir:
        candidates.append(Path(cache_dir) / ".copernicusmarine-credentials")
    for cand in candidates:
        if cand.exists():
            log.info("using copernicusmarine login cache at %s", cand)
            return None, None

    raise CredentialsError(
        "Copernicus Marine credentials not found. Run `copernicusmarine login` "
        "once, or set COPERNICUSMARINE_SERVICE_USERNAME and "
        "COPERNICUSMARINE_SERVICE_PASSWORD. Free registration: "
        "https://data.marine.copernicus.eu/register"
    )


def earthdata_netrc() -> Path:
    """Verify a ~/.netrc entry for `urs.earthdata.nasa.gov` (PODAAC SST/wind/color).

    Returns the netrc path. Raises with the CRW no-auth fallback hint if missing.
    """
    netrc_path = Path(os.environ.get("NETRC", str(Path.home() / ".netrc")))
    machine = "urs.earthdata.nasa.gov"
    if netrc_path.exists() and machine in netrc_path.read_text():
        return netrc_path
    raise CredentialsError(
        f"NASA Earthdata login not found. Add a '{machine}' entry to {netrc_path} "
        "(register free at https://urs.earthdata.nasa.gov), or use the no-auth "
        "NOAA CoastWatch ERDDAP fallback by setting products.sst_dataset='CRW'."
    )


def ee_project() -> str:
    """Earth Engine project id from env `EE_PROJECT` (CoastSat / shoreline only)."""
    proj = os.environ.get("EE_PROJECT")
    if not proj:
        raise CredentialsError(
            "Earth Engine project not set. Run `earthengine authenticate` and set "
            "EE_PROJECT=<your-gcp-project> (required only for shoreline extraction)."
        )
    return proj
