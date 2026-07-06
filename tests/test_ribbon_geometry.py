"""The ribbon must be flyable: no corner may exceed the plane's turn
physics. Every consecutive segment pair in the polyline is checked for
heading change; arcs sampled at ~10° per step mean anything sharply
above that is a corner the plane cannot track at its bank limit.
"""
from __future__ import annotations

import math
import unittest

from uav.nav.flight_plan_v2 import plan_path, pick_cruise_alt_agl, _turn_radius_nm
from uav.nav.geo import bearing_deg, haversine_m


def _max_corner_deg(points, skip_ground: bool = True) -> float:
    """Largest heading change between consecutive polyline segments.
    Ground/rollout points are excluded (taxi turns are not flight)."""
    worst = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        if skip_ground and (a.phase in ("GROUND", "ROLLOUT")
                            or c.phase in ("GROUND", "ROLLOUT")):
            continue
        # Ignore near-coincident points (numerical noise).
        if (haversine_m(a.lat, a.lon, b.lat, b.lon) < 30.0
                or haversine_m(b.lat, b.lon, c.lat, c.lon) < 30.0):
            continue
        h1 = bearing_deg(a.lat, a.lon, b.lat, b.lon)
        h2 = bearing_deg(b.lat, b.lon, c.lat, c.lon)
        turn = abs(((h2 - h1 + 180.0) % 360.0) - 180.0)
        worst = max(worst, turn)
    return worst


def _plan(dep_hdg: float, dest_lat: float = 47.40, dest_lon: float = -94.77):
    return plan_path(
        dep_lat=45.42, dep_lon=-91.77, dep_alt_ft=1100.0,
        dep_heading=dep_hdg,
        dest_lat=dest_lat, dest_lon=dest_lon, dest_alt_ft=1380.0,
        dest_rwy_heading=270.0,
        dest_threshold_lat=dest_lat, dest_threshold_lon=dest_lon,
        cruise_alt_ft=5000.0, v_stall=98.4, v_cruise_kts=132.7,
    )


class TestNoSharpCorners(unittest.TestCase):
    # 35°: arcs sample at ~10°/step; genuine corners used to show 60-120°.
    LIMIT_DEG = 35.0

    def test_aligned_departure(self):
        r = _plan(dep_hdg=300.0)  # roughly toward destination
        self.assertLess(_max_corner_deg(r.points), self.LIMIT_DEG)

    def test_perpendicular_departure(self):
        r = _plan(dep_hdg=30.0)
        self.assertLess(_max_corner_deg(r.points), self.LIMIT_DEG)

    def test_course_reversal_departure(self):
        # Taking off pointing AWAY from the destination — the ribbon
        # must open with a half-circle, not an instant U-turn.
        r = _plan(dep_hdg=120.0)
        self.assertLess(_max_corner_deg(r.points), self.LIMIT_DEG)

    def test_turn_radius_scales_with_speed(self):
        self.assertGreater(_turn_radius_nm(180.0), _turn_radius_nm(100.0))
        # ~0.43 nm at SF50 cruise
        self.assertAlmostEqual(_turn_radius_nm(132.7), 0.43, delta=0.05)


class TestCruiseAltPicker(unittest.TestCase):
    def test_short_hop_stays_low(self):
        self.assertEqual(pick_cruise_alt_agl(9.0), 1500.0)

    def test_long_trip_flies_high(self):
        self.assertEqual(pick_cruise_alt_agl(30.0), 5000.0)

    def test_ramp_is_monotonic(self):
        alts = [pick_cruise_alt_agl(d) for d in (5, 10, 14, 18, 22, 30)]
        self.assertEqual(alts, sorted(alts))


if __name__ == "__main__":
    unittest.main()
