"""Local data-file locations for the demo/script layer.

The OSM PBF and Porto Kaggle CSV are large external inputs the repo does not
ship. Their paths resolve from environment variables, falling back to the
conventional local layout under ``~/Documents/shared_data``:

    GPS_RECON_DATA   base directory (default: ~/Documents/shared_data)
    GPS_RECON_PBF    full path to the Portugal OSM PBF (overrides the base)
    GPS_RECON_CSV    full path to the Porto train.csv (overrides the base)

Resolution is lazy (at call time, not import time) so a script may set
``os.environ`` before calling — matching the ``GPS_RECON_BBOX_*`` convention
used elsewhere in this layer.
"""

from __future__ import annotations

import os
from pathlib import Path


def shared_data_root() -> Path:
    """Root of the local OSM/Porto data tree. Override with ``$GPS_RECON_DATA``."""
    return Path(os.environ.get("GPS_RECON_DATA") or Path.home() / "Documents" / "shared_data")


def osm_pbf_path() -> Path:
    """Portugal OSM PBF path. Override with ``$GPS_RECON_PBF``."""
    env = os.environ.get("GPS_RECON_PBF")
    return Path(env) if env else shared_data_root() / "osm" / "portugal-latest.osm.pbf"


def porto_csv_path() -> Path:
    """Porto Kaggle ``train.csv`` path. Override with ``$GPS_RECON_CSV``."""
    env = os.environ.get("GPS_RECON_CSV")
    return Path(env) if env else shared_data_root() / "porto" / "train.csv"
