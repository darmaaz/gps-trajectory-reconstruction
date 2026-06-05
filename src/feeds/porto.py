"""Porto taxi trajectory feed (Kaggle ECML/PKDD 2015 challenge).

The Porto dataset is one row per *trip*, with the trip's GPS pings packed
into a JSON-encoded `POLYLINE` column at a fixed 15-second cadence. This
module is the boundary between that storage layout and the pipeline's
per-ping `RawObservation` contract — Porto-specific assumptions (column
names, polyline format, 15s sampling, the `MISSING_DATA` flag) live here
so the core pipeline stays data-source-agnostic.

CSV schema (per Kaggle):
    TRIP_ID       string, unique
    CALL_TYPE     A/B/C
    ORIGIN_CALL   nullable int
    ORIGIN_STAND  nullable int
    TAXI_ID       int
    TIMESTAMP     unix seconds, trip start
    DAY_TYPE      A/B/C
    MISSING_DATA  bool — True means the polyline has gaps; we drop these
    POLYLINE      JSON string of [[lon, lat], [lon, lat], ...]

Two entry points:

    `iter_porto_trips`  — generator yielding `(trip_id, list[RawObservation])`
                          per trip. Streams the CSV in chunks; never holds
                          more than `chunksize` rows in memory at once.

    `polyline_to_observations` — single-trip helper that explodes one
                          polyline string into a `RawObservation` list at
                          15s intervals from `start_unix`. Pure function;
                          easy to unit-test.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..model import RawObservation

PORTO_SAMPLING_S: int = 15


def polyline_to_observations(
    polyline: str | list[list[float]],
    start_unix: int | float,
) -> list[RawObservation]:
    """Explode a Porto polyline into per-ping `RawObservation`s.

    `polyline` is either the raw JSON string from the CSV or an
    already-parsed `[[lon, lat], ...]` list. Each entry becomes one
    observation; timestamps step by `PORTO_SAMPLING_S` seconds from
    `start_unix`. Returns `[]` for empty/malformed polylines so callers
    can filter without try/except.
    """
    if isinstance(polyline, str):
        if not polyline or polyline == "[]":
            return []
        try:
            coords = json.loads(polyline)
        except json.JSONDecodeError:
            return []
    else:
        coords = polyline

    if not coords:
        return []

    t0 = float(start_unix)
    out: list[RawObservation] = []
    for i, pair in enumerate(coords):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        lon, lat = float(pair[0]), float(pair[1])
        ts = datetime.fromtimestamp(
            t0 + i * PORTO_SAMPLING_S, tz=timezone.utc,
        )
        out.append(RawObservation(
            timestamp=ts, lat=lat, lon=lon, reported_speed=None,
        ))
    return out


def iter_porto_trips(
    csv_path: Path,
    *,
    chunksize: int = 10_000,
    skip_missing: bool = True,
    min_pings: int = 2,
) -> Iterator[tuple[str, list[RawObservation]]]:
    """Stream Porto trips as `(trip_id, list[RawObservation])` pairs.

    Parameters
    ----------
    csv_path : path to the Porto CSV (gzip is auto-handled by pandas).
    chunksize : rows per pandas chunk; the loader holds at most this many
        rows in memory at once. The default keeps memory low on the full
        1.7M-trip file.
    skip_missing : drop rows where `MISSING_DATA` is True (Kaggle's flag
        for trips with known polyline gaps).
    min_pings : drop trips with fewer than this many pings after explosion.
        Inference accepts ≥1, but trips with only one ping yield emission-
        only marginals — usually not interesting for sanity runs.

    Yields one `(trip_id, observations)` pair per surviving trip, in CSV
    order.
    """
    cols = ["TRIP_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"]
    reader = pd.read_csv(
        csv_path, usecols=cols, chunksize=chunksize,
        dtype={"TRIP_ID": str, "TIMESTAMP": "int64", "POLYLINE": str},
    )
    for chunk in reader:
        if skip_missing and "MISSING_DATA" in chunk.columns:
            chunk = chunk[~chunk["MISSING_DATA"].astype(bool)]
        for row in chunk.itertuples(index=False):
            obs = polyline_to_observations(row.POLYLINE, row.TIMESTAMP)
            if len(obs) >= min_pings:
                yield row.TRIP_ID, obs
