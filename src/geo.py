"""Geographic helpers — distance, bearing, local equirectangular projection.

The pipeline operates on metres, not degrees. These helpers convert at three
boundaries: ingest (parquet → metric kinematics), the network boundary
(point ↔ edge projection), and any human-readable export. Local
equirectangular is the working frame — sub-100 km extents at Mexico latitudes
introduce negligible distortion versus full UTM.
"""

from __future__ import annotations

import numpy as np

EARTH_R_M: float = 6_371_000.0
M_PER_DEG_LAT: float = 111_320.0


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres. Element-wise on numpy arrays."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def forward_azimuth_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees (0 = north, clockwise). Element-wise."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def equirectangular_distance_m(lat_query, lon_query, lat_target, lon_target):
    """Distance in metres between (lat, lon) pairs via local equirectangular
    scaling at the *query* latitude. Sub-kilometre accurate; cheaper than
    haversine. Used inside nearest-edge projection, where query and target are
    by construction nearby.
    """
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(lat_query))
    dx = (np.asarray(lon_query) - np.asarray(lon_target)) * m_per_deg_lon
    dy = (np.asarray(lat_query) - np.asarray(lat_target)) * M_PER_DEG_LAT
    return np.sqrt(dx * dx + dy * dy)
