"""Shared pytest fixtures.

Synthetic + grid road network used across multiple test modules:

         N3 (19.435, -99.130)
         |
  N1 ---N0--- N2          and   N1 ---9---> N4   (residential diagonal)
 (-99.135) (-99.130) (-99.125)
         |
         N4 (19.425, -99.130)

Each direction is a separate edge so directed routing has a real choice.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytest
from shapely.geometry import LineString

from src.network import RoadNetwork, build_network_from_records


def _edge(eid: int, rc: str, kmh: float, a: int, b: int,
          lat_a: float, lon_a: float, lat_b: float, lon_b: float):
    geom = LineString([(lon_a, lat_a), (lon_b, lat_b)])
    speed_ms = kmh / 3.6
    dlat_m = (lat_b - lat_a) * 111_320.0
    dlon_m = (lon_b - lon_a) * 111_320.0 * math.cos(math.radians(lat_a))
    length = math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)
    return (eid, rc, speed_ms, geom, a, b, length)


@pytest.fixture(scope="session")
def grid_network() -> RoadNetwork:
    records = [
        _edge(1, "primary",     60, 0, 1, 19.430, -99.130, 19.430, -99.135),
        _edge(2, "primary",     60, 1, 0, 19.430, -99.135, 19.430, -99.130),
        _edge(3, "primary",     60, 0, 2, 19.430, -99.130, 19.430, -99.125),
        _edge(4, "primary",     60, 2, 0, 19.430, -99.125, 19.430, -99.130),
        _edge(5, "secondary",   50, 0, 3, 19.430, -99.130, 19.435, -99.130),
        _edge(6, "secondary",   50, 3, 0, 19.435, -99.130, 19.430, -99.130),
        _edge(7, "secondary",   50, 0, 4, 19.430, -99.130, 19.425, -99.130),
        _edge(8, "secondary",   50, 4, 0, 19.425, -99.130, 19.430, -99.130),
        _edge(9, "residential", 30, 1, 4, 19.430, -99.135, 19.425, -99.130),
    ]
    return build_network_from_records(records)


@pytest.fixture
def t0() -> datetime:
    return datetime(2026, 5, 5, 12, 0, 0)
