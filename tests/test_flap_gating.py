"""Situational flap control: flaps are lift, deployed only when needed.

On the glideslope, flaps deploy when the plane has sunk ≥100 ft BELOW
the slope (their lift pulls it back up) and retract to clean when back
within 50 ft of the slope or above it (speed permitting). On-slope or
high, the wing stays clean — deploying flaps while high balloons the
plane further off the slope and it floats past the touchdown point.
"""
from __future__ import annotations

import unittest

from uav.core.flight_engine import (
    FlightEngine, _FLAP_DEPLOY_BELOW_FT, _FLAP_CLEAN_BELOW_FT,
)
from uav.nav.flight_plan_v2 import plan_path
from uav.sim.types import Telemetry, Targets


def _telem(alt_ft: float, lat: float, lon: float, spd: float = 110.0):
    return Telemetry(
        airspeed_kts=spd, altitude_ft=alt_ft, pitch_deg=-3.0, roll_deg=0.0,
        heading_deg=270.0, timestamp=0.0, lat_deg=lat, lon_deg=lon,
        agl_m=(alt_ft - 1380.0) / 3.28084, vs_fpm=-800.0,
    )


class TestFlapControl(unittest.TestCase):
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

    def _cmd_alt(self) -> float:
        """Commanded glideslope alt at the test position."""
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=3000.0, airspeed_kts=110.0,
            flap_ratio=0.0, gear_down=False,
        )
        t = _telem(3000.0, self.lat, self.lon)
        return self.engine._resolve(self.kf_flap, t, self.ribbon).altitude_ft

    def _resolve(self, alt_ft: float, prev_flap: float, spd: float = 110.0):
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=alt_ft, airspeed_kts=spd,
            flap_ratio=prev_flap, gear_down=False,
        )
        t = _telem(alt_ft, self.lat, self.lon, spd=spd)
        return self.engine._resolve(self.kf_flap, t, self.ribbon)

    def test_flaps_deploy_when_sunk_below_slope(self):
        cmd = self._cmd_alt()
        out = self._resolve(cmd - _FLAP_DEPLOY_BELOW_FT - 20.0, prev_flap=0.0)
        self.assertEqual(out.flap_ratio, 0.5,
                         "≥100 ft below slope → flaps deploy (need the lift).")

    def test_flaps_stay_clean_on_slope(self):
        cmd = self._cmd_alt()
        out = self._resolve(cmd, prev_flap=0.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "On the slope, no flaps needed.")

    def test_flaps_stay_clean_when_high(self):
        cmd = self._cmd_alt()
        out = self._resolve(cmd + 300.0, prev_flap=0.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "Above the slope, flaps would balloon us higher.")

    def test_flaps_retract_when_back_on_slope(self):
        cmd = self._cmd_alt()
        out = self._resolve(cmd, prev_flap=0.5, spd=110.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "Back on slope with speed in hand → clean up.")

    def test_flaps_do_not_retract_when_slow(self):
        cmd = self._cmd_alt()
        v_app = self.ribbon.geometry.v_approach
        out = self._resolve(cmd, prev_flap=0.5, spd=v_app - 5.0)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Never retract flaps below v_approach — stall risk.")

    def test_flaps_hold_in_hysteresis_band(self):
        cmd = self._cmd_alt()
        mid = (_FLAP_DEPLOY_BELOW_FT + _FLAP_CLEAN_BELOW_FT) / 2.0
        out = self._resolve(cmd - mid, prev_flap=0.5)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Between thresholds the current setting holds.")

    def test_no_deploy_when_fast_even_below_slope(self):
        # THE crash case: below the slope (altitude logic wants flaps)
        # but going fast. Speed protection must win — flaps at speed
        # balloon the plane and the recovery dive hits the ground.
        cmd = self._cmd_alt()
        flap_safe = self.ribbon.geometry.flap_safe_kts
        out = self._resolve(cmd - 200.0, prev_flap=0.0, spd=flap_safe + 15.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "No flap deployment above flap_safe speed. Ever.")

    def test_overspeed_forces_retraction(self):
        cmd = self._cmd_alt()
        flap_safe = self.ribbon.geometry.flap_safe_kts
        out = self._resolve(cmd - 200.0, prev_flap=0.5,
                            spd=flap_safe * 1.08 + 5.0)
        self.assertEqual(out.flap_ratio, 0.0,
                         "Hard overspeed pulls deployed flaps back in.")

    def test_fast_flare_entry_does_not_slam_full_flaps(self):
        # Slightly over flap_safe (inside the hysteresis band): the
        # scheduled full flaps are blocked, current setting kept.
        # Well past 1.08×flap_safe the hard retract takes over instead.
        kf_flare = next(k for k in self.ribbon.keyframes
                        if k.name == "FLARE")
        flap_safe = self.ribbon.geometry.flap_safe_kts
        spd = flap_safe + 5.0  # over the gate, under 1.08× hard limit
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=1420.0,
            airspeed_kts=spd, flap_ratio=0.5, gear_down=True,
        )
        t = _telem(1410.0, 47.40, -94.7705, spd=spd)
        out = self.engine._resolve(kf_flare, t, self.ribbon)
        self.assertEqual(out.flap_ratio, 0.5,
                         "Fast flare entry keeps current flaps, no full slam.")

    def test_flare_exempt(self):
        kf_flare = next(k for k in self.ribbon.keyframes
                        if k.name == "FLARE")
        self.engine._prev_targets = Targets(
            heading_deg=270.0, altitude_ft=1420.0, airspeed_kts=100.0,
            flap_ratio=0.5, gear_down=True,
        )
        t = _telem(1410.0, 47.40, -94.7705, spd=100.0)
        out = self.engine._resolve(kf_flare, t, self.ribbon)
        self.assertEqual(out.flap_ratio, 1.0,
                         "FLARE always gets full flaps.")


if __name__ == "__main__":
    unittest.main()
