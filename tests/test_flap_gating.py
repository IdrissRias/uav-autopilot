"""Situational flap deployment.

Flaps deploying while the plane is above the glideslope balloon it
further off the slope (pitch tracks speed on descent, throttle is near
idle — nothing counters the added lift) and the plane floats past the
touchdown point. The engine defers scheduled flap increases until the
plane is within _FLAP_DEFER_ABOVE_FT of the commanded slope altitude.
"""
from __future__ import annotations

import unittest

from uav.core.flight_engine import FlightEngine, _FLAP_DEFER_ABOVE_FT
from uav.nav.flight_plan_v2 import plan_path
from uav.sim.types import Telemetry, Targets


def _telem(alt_ft: float, lat: float, lon: float, spd: float = 110.0):
    return Telemetry(
        airspeed_kts=spd, altitude_ft=alt_ft, pitch_deg=-3.0, roll_deg=0.0,
        heading_deg=270.0, timestamp=0.0, lat_deg=lat, lon_deg=lon,
        agl_m=(alt_ft - 1380.0) / 3.28084, vs_fpm=-800.0,
    )


class TestFlapGating(unittest.TestCase):
    def setUp(self):
        self.ribbon = plan_path(
            dep_lat=45.42, dep_lon=-91.77, dep_alt_ft=1100.0,
            dep_heading=10.0,
            dest_lat=47.40, dest_lon=-94.77, dest_alt_ft=1380.0,
            dest_rwy_heading=270.0,
            dest_threshold_lat=47.40, dest_threshold_lon=-94.77,
            cruise_alt_ft=5000.0, v_stall=98.4,
        )
        self.engine = FlightEngine({})
        self.kf_flap = next(k for k in self.ribbon.keyframes
                            if k.name == "DESCENT_FLAP")
        # Plane ~2 nm east of threshold, on the approach side.
        self.lat, self.lon = 47.40, -94.72

    def _resolve_at(self, alt_ft: float):
        # Previous tick had clean config (flaps up).
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=alt_ft, airspeed_kts=110.0,
            flap_ratio=0.0, gear_down=False,
        )
        t = _telem(alt_ft, self.lat, self.lon)
        return self.engine._resolve(self.kf_flap, t, self.ribbon)

    def test_flaps_deferred_when_high_above_slope(self):
        # First resolve on-slope to learn the commanded alt at this spot.
        on_slope = self._resolve_at(3000.0).altitude_ft
        high = self._resolve_at(on_slope + _FLAP_DEFER_ABOVE_FT + 150.0)
        self.assertEqual(
            high.flap_ratio, 0.0,
            "Flap increase should be DEFERRED while the plane is well "
            "above the glideslope — deploying would balloon it higher."
        )

    def test_flaps_deploy_when_on_slope(self):
        on_slope_alt = self._resolve_at(3000.0).altitude_ft
        on = self._resolve_at(on_slope_alt)  # exactly on slope
        self.assertEqual(
            on.flap_ratio, 0.5,
            "Scheduled flaps should deploy once the plane is on the slope."
        )

    def test_flaps_never_retract_once_out(self):
        # Previous tick already had half flaps; plane is high. Gating
        # must not pull deployed flaps back in.
        cmd_alt = self._resolve_at(3000.0).altitude_ft
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=cmd_alt, airspeed_kts=110.0,
            flap_ratio=0.5, gear_down=False,
        )
        t = _telem(cmd_alt + 400.0, self.lat, self.lon)
        out = self.engine._resolve(self.kf_flap, t, self.ribbon)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Deployed flaps must never retract mid-approach.")

    def test_flare_exempt_from_gating(self):
        kf_flare = next(k for k in self.ribbon.keyframes
                        if k.name == "FLARE")
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=1420.0, airspeed_kts=100.0,
            flap_ratio=0.5, gear_down=True,
        )
        # Threshold-ish position, 30 ft AGL. Commanded alt clamps near
        # the pavement so the plane always reads "high" — FLARE must
        # still get its full flaps.
        t = _telem(1410.0, 47.40, -94.7705, spd=100.0)
        out = self.engine._resolve(kf_flare, t, self.ribbon)
        self.assertEqual(out.flap_ratio, 1.0,
                         "FLARE is exempt from flap gating.")


if __name__ == "__main__":
    unittest.main()
