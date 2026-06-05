"""Observation dataclasses — inputs and preprocessing intermediates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RawObservation:
    """One GPS report at the pipeline boundary. Fields beyond
    (timestamp, lat, lon) come from the upstream feed if exposed; absent
    in feeds (e.g. Porto Kaggle) that don't carry per-ping metadata.
    """

    timestamp: datetime
    lat: float
    lon: float
    fix_type: str | None = None
    hdop: float | None = None
    reported_speed: float | None = None    # m/s, normalised at ingest


@dataclass(frozen=True)
class CollapsedObservation:
    """Output of preprocessing (hygiene → spike-removal → replay-collapse
    → collapse-by-uniqueness → stale-jump detection). `t_first` is the
    canonical timestamp consumed by inference; `t_last - t_first` on a
    non-stale run is the confirmed dwell at this position.

    `reported_speed_max_ms` is the maximum `reported_speed` observed across
    the raw pings absorbed into this run (None if no raw ping carried a
    speed reading). It's used by replay-burst detection to recognise the
    frozen-chip pattern — multiple raw pings at the same position while
    the device kept reporting non-zero speed.

    `dropped_during_replay` is set by `preprocessing.replay_detection.
    drop_replay_bursts` for observations identified as members of a
    buffered-replay burst. The orchestrator filters them out before
    inference; they remain in the post-collapse list for observability.
    """

    lat: float
    lon: float
    t_first: datetime
    t_last: datetime
    collapsed_count: int
    stale_flagged: bool = False
    dropped_during_replay: bool = False
    reported_speed_max_ms: float | None = None
