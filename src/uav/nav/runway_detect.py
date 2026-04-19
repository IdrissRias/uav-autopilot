"""Peregrine — Runway detection.

Given the aircraft's GPS position and heading, determine if it's on a
runway and which one. Uses the RunwayBox geometry from runway_box.py
and runway data from the local SQLite database.

Returns a RunwayDetection with airport ICAO, runway designator, position
on the runway, and confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .geo import haversine_m
from .runway_box import RunwayBox, RunwayPosition, runway_box_from_db


# How close to a runway centerline the plane must be (meters)
MAX_LATERAL_M = 30.0  # ~100ft — wider than most runways
MAX_LATERAL_FT = MAX_LATERAL_M * 3.281

# How close heading must match runway heading (degrees)
MAX_HEADING_DIFF_DEG = 25.0

# How far from an airport to even bother checking (meters)
MAX_AIRPORT_RANGE_M = 5000.0  # ~2.7nm


@dataclass
class RunwayDetection:
    """Result of runway detection."""
    detected: bool
    airport_icao: str = ""
    airport_name: str = ""
    runway_designator: str = ""
    runway_heading_deg: float = 0.0
    runway_length_ft: float = 0.0
    position: Optional[RunwayPosition] = None
    heading_match_deg: float = 0.0  # how close our heading matches

    @property
    def summary(self) -> str:
        if not self.detected:
            return "Not on a runway"
        return (f"On RWY {self.runway_designator} at {self.airport_icao} "
                f"({self.airport_name}), {self.runway_length_ft:.0f}ft, "
                f"hdg {self.runway_heading_deg:.0f}°")


def _wrap_heading_diff(a: float, b: float) -> float:
    """Smallest signed difference between two headings."""
    d = (a - b + 180) % 360 - 180
    return abs(d)


def _runway_heading_for_designator(designator: str, db_heading: float) -> float:
    """Get the actual heading for a specific runway end.

    Runway designators like '13/31' represent two directions.
    The db_heading is for the first number. The reciprocal is +180.
    """
    parts = designator.split("/")
    if len(parts) == 2:
        return db_heading, (db_heading + 180) % 360
    return (db_heading,)


def detect_runway(
    lat: float,
    lon: float,
    heading_deg: float,
    airports: Optional[List[dict]] = None,
) -> RunwayDetection:
    """Detect if the aircraft is on a runway.

    Args:
        lat, lon: Aircraft GPS position
        heading_deg: Aircraft magnetic heading
        airports: Pre-loaded airport list (with runways). If None, loads from DB.

    Returns:
        RunwayDetection with the best match, or detected=False.
    """
    if airports is None:
        from ..db import local_db
        airports = []
        for apt in local_db.list_airports():
            full_apt = local_db.get_airport(apt["icao_code"])
            if full_apt:
                airports.append(full_apt)

    best_match: Optional[RunwayDetection] = None
    best_score = float("inf")

    for airport in airports:
        # Quick distance check — skip airports that are far away
        dist_m = haversine_m(lat, lon, airport["lat"], airport["lon"])
        if dist_m > MAX_AIRPORT_RANGE_M:
            continue

        for runway in airport.get("runways", []):
            box = runway_box_from_db(runway)
            pos = box.locate(lat, lon)

            # Must be within the runway rectangle (with some lateral tolerance)
            if not (-0.1 <= pos.pct_used <= 1.1):  # allow slight overshoot
                continue
            if abs(pos.lateral_ft) > MAX_LATERAL_FT:
                continue

            # Check heading alignment with either end of the runway
            db_heading = runway["heading_deg"]
            designator = runway.get("designator", "")
            headings = _runway_heading_for_designator(designator, db_heading)

            best_hdg_diff = min(_wrap_heading_diff(heading_deg, h) for h in headings)
            if best_hdg_diff > MAX_HEADING_DIFF_DEG:
                continue

            # Which designator are we facing?
            # If heading closer to db_heading → first designator, else reciprocal
            parts = designator.split("/")
            if len(parts) == 2:
                if _wrap_heading_diff(heading_deg, db_heading) <= _wrap_heading_diff(heading_deg, (db_heading + 180) % 360):
                    facing_designator = parts[0]
                    facing_heading = db_heading
                else:
                    facing_designator = parts[1]
                    facing_heading = (db_heading + 180) % 360
            else:
                facing_designator = designator
                facing_heading = db_heading

            # Score: lower is better (prefer closer to centerline + better heading match)
            score = abs(pos.lateral_ft) + best_hdg_diff * 10

            if score < best_score:
                best_score = score
                best_match = RunwayDetection(
                    detected=True,
                    airport_icao=airport["icao_code"],
                    airport_name=airport["name"],
                    runway_designator=facing_designator,
                    runway_heading_deg=facing_heading,
                    runway_length_ft=runway["length_ft"],
                    position=pos,
                    heading_match_deg=best_hdg_diff,
                )

    return best_match or RunwayDetection(detected=False)
