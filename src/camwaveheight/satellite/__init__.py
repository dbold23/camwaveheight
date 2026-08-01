"""Satellite + model ocean data stream for a site.

A sibling to the cam pipeline: pulls satellite / model wave, shoreline, and
environmental products for a site's location and time windows, caches them like
`camwaveheight.cdip`, and feeds them into the same validation machinery
(`camwaveheight.validate`) as additional ground truth.

The motion-energy Hs pipeline remains the system of record; nothing here writes
to the cam's regression coefficients.

Submodules (each imports its heavy/optional deps lazily):
  common       — UTC/cache/bbox/haversine helpers shared across fetchers
  creds        — credential resolution (CMEMS / Earthdata / Earth Engine)
  waves        — CMEMS model + reanalysis + altimeter L3 SWH  (cdip.py twin)
  environment  — SST / wind / ocean-color panels
  shoreline    — CoastSat satellite-derived shoreline + transects

This package's __init__ stays import-light on purpose; import the submodule you
need (e.g. `from camwaveheight.satellite import waves`).
"""

from __future__ import annotations

__all__ = ["common", "creds", "waves", "environment", "shoreline"]
