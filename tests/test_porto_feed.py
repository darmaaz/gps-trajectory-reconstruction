"""Tests for the Porto Kaggle CSV feed adapter."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.feeds.porto import (
    PORTO_SAMPLING_S,
    iter_porto_trips,
    polyline_to_observations,
)


class TestPolylineToObservations:
    def test_explodes_each_pair_into_an_observation(self) -> None:
        polyline = "[[-8.61,41.14],[-8.62,41.15],[-8.63,41.16]]"
        obs = polyline_to_observations(polyline, start_unix=1_372_636_858)
        assert len(obs) == 3

    def test_timestamps_step_by_15_seconds(self) -> None:
        polyline = "[[-8.61,41.14],[-8.62,41.15],[-8.63,41.16]]"
        obs = polyline_to_observations(polyline, start_unix=1_372_636_858)
        deltas = [
            (obs[i + 1].timestamp - obs[i].timestamp).total_seconds()
            for i in range(len(obs) - 1)
        ]
        assert deltas == [PORTO_SAMPLING_S, PORTO_SAMPLING_S]
        assert obs[0].timestamp == datetime.fromtimestamp(
            1_372_636_858, tz=timezone.utc,
        )

    def test_lon_lat_order_in_polyline_maps_to_lat_lon_in_obs(self) -> None:
        # Porto stores [lon, lat]; RawObservation is (lat, lon). The adapter
        # is responsible for the swap.
        polyline = "[[-8.61, 41.14]]"
        obs = polyline_to_observations(polyline, start_unix=0)
        assert obs[0].lat == pytest.approx(41.14)
        assert obs[0].lon == pytest.approx(-8.61)

    def test_reported_speed_is_none(self) -> None:
        # Porto has no speed column; downstream code must tolerate None.
        polyline = "[[-8.61, 41.14]]"
        obs = polyline_to_observations(polyline, start_unix=0)
        assert obs[0].reported_speed is None

    @pytest.mark.parametrize("bad", ["", "[]", "not-json", "{not a list}"])
    def test_empty_or_malformed_returns_empty_list(self, bad: str) -> None:
        assert polyline_to_observations(bad, start_unix=0) == []

    def test_accepts_already_parsed_list(self) -> None:
        # Convenience for callers that have already deserialized.
        coords = [[-8.61, 41.14], [-8.62, 41.15]]
        obs = polyline_to_observations(coords, start_unix=0)
        assert len(obs) == 2


class TestIterPortoTrips:
    @pytest.fixture
    def csv_path(self, tmp_path: Path) -> Path:
        # Three trips: a normal one, a MISSING_DATA one, and an empty polyline.
        df = pd.DataFrame({
            "TRIP_ID": ["T1", "T2", "T3", "T4"],
            "CALL_TYPE": ["A", "A", "A", "A"],
            "ORIGIN_CALL": [None, None, None, None],
            "ORIGIN_STAND": [None, None, None, None],
            "TAXI_ID": [20000001, 20000002, 20000003, 20000004],
            "TIMESTAMP": [1_372_636_858, 1_372_636_900, 1_372_637_000, 1_372_637_100],
            "DAY_TYPE": ["A", "A", "A", "A"],
            "MISSING_DATA": [False, True, False, False],
            "POLYLINE": [
                "[[-8.61,41.14],[-8.62,41.15],[-8.63,41.16]]",
                "[[-8.61,41.14],[-8.62,41.15]]",
                "[]",
                "[[-8.61,41.14],[-8.62,41.15],[-8.63,41.16],[-8.64,41.17]]",
            ],
        })
        path = tmp_path / "porto.csv"
        df.to_csv(path, index=False)
        return path

    def test_yields_only_valid_trips(self, csv_path: Path) -> None:
        trips = list(iter_porto_trips(csv_path))
        ids = [t[0] for t in trips]
        assert ids == ["T1", "T4"]

    def test_yields_correct_observation_counts(self, csv_path: Path) -> None:
        trips = dict(iter_porto_trips(csv_path))
        assert len(trips["T1"]) == 3
        assert len(trips["T4"]) == 4

    def test_min_pings_filter(self, csv_path: Path) -> None:
        trips = list(iter_porto_trips(csv_path, min_pings=4))
        assert [t[0] for t in trips] == ["T4"]

    def test_is_a_generator_not_a_list(self, csv_path: Path) -> None:
        # Confirms the adapter streams rather than materializing the whole
        # file. Important for the 1.7M-trip production CSV.
        import types
        result = iter_porto_trips(csv_path)
        assert isinstance(result, types.GeneratorType)
