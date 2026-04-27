"""Guard against structural-limit violations in the descent cascade.

Flight 20260419_135940 deployed flaps at 170 kts and gear at 166 kts
because DESCENT_FLAP and DESCENT_GEAR triggers were ANY(speed_lte,
agl_lte) — the AGL condition alone was enough to fire the transition.
These tests assert the triggers now require BOTH speed AND altitude.
"""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from uav.nav.flight_plan_v2 import plan_path


def _telem(spd_kts: float, alt_ft: float, lat: float, lon: float):
    """Minimal Telemetry-shaped object for Trigger.fired()."""
    return SimpleNamespace(
        airspeed_kts=spd_kts,
        altitude_ft=alt_ft,
        lat_deg=lat,
        lon_deg=lon,
        has_position=lambda: True,
    )


class TestDescentCascadeTriggers(unittest.TestCase):
    """Prevent the 'cascade-through-in-one-tick' bug from regressing."""

    def setUp(self):
        # KRPD (dep) → KUBE (arr), the same route as flight 135940.
        self.ribbon = plan_path(
            dep_lat=45.4236, dep_lon=-91.7730, dep_alt_ft=1100.0,
            dep_heading=10.0,
            dest_lat=47.3972, dest_lon=-94.7717, dest_alt_ft=1380.0,
            dest_rwy_heading=270.0,
            dest_threshold_lat=47.3972, dest_threshold_lon=-94.7717,
            cruise_alt_ft=5000.0,
            v_stall=77.0,
        )
        self.by_name = {k.name: k for k in self.ribbon.keyframes}

    # ── DESCENT → DESCENT_FLAP ───────────────────────────────────────

    def test_flaps_do_not_deploy_at_170_kts(self):
        """Observed failure: flight 135940 deployed flaps at 172 kts."""
        kf = self.by_name["DESCENT"]
        # Low AGL but over Vfe — trigger must NOT fire.
        t = _telem(170.0, 2100.0, 47.0, -94.5)
        self.assertFalse(
            kf.trigger.fired(t, agl_ft=1800.0),
            "DESCENT→DESCENT_FLAP fired with speed=170kts > Vfe "
            "(flap_safe = 1.50×Vs = ~115kts). Flaps would exceed Vfe."
        )

    def test_flaps_do_deploy_when_slow_and_low(self):
        """Positive case: both conditions met → flaps come out."""
        kf = self.by_name["DESCENT"]
        t = _telem(110.0, 2100.0, 47.0, -94.5)  # below Vfe, below 2000 AGL
        self.assertTrue(
            kf.trigger.fired(t, agl_ft=1900.0),
            "DESCENT→DESCENT_FLAP should fire when spd<Vfe AND agl<2000ft."
        )

    def test_flaps_do_not_deploy_if_high_even_when_slow(self):
        """At cruise alt, even slow, we shouldn't deploy flaps yet."""
        kf = self.by_name["DESCENT"]
        t = _telem(100.0, 5000.0, 47.0, -94.5)
        self.assertFalse(
            kf.trigger.fired(t, agl_ft=4000.0),
            "DESCENT→DESCENT_FLAP should NOT fire at 4000ft AGL."
        )

    # ── DESCENT_FLAP → DESCENT_GEAR ──────────────────────────────────

    def test_gear_does_not_drop_at_166_kts(self):
        """Observed failure: flight 135940 dropped gear at 166 kts."""
        kf = self.by_name["DESCENT_FLAP"]
        t = _telem(166.0, 1500.0, 47.0, -94.6)
        self.assertFalse(
            kf.trigger.fired(t, agl_ft=1000.0),
            "DESCENT_FLAP→DESCENT_GEAR fired with speed=166kts > Vlo "
            "(gear_safe = 1.30×Vs = ~100kts)."
        )

    def test_gear_drops_when_slow_and_low(self):
        kf = self.by_name["DESCENT_FLAP"]
        t = _telem(95.0, 1500.0, 47.0, -94.6)
        self.assertTrue(
            kf.trigger.fired(t, agl_ft=1100.0),
            "DESCENT_FLAP→DESCENT_GEAR should fire when spd<Vlo AND agl<1200ft."
        )


class TestDecelZoneLength(unittest.TestCase):
    """The decel zone must be large enough to actually bleed speed.
    A jet coasting at idle loses ~1 kt/sec; we need seconds of idle
    coast to go from Vcruise (~154 kts) to Vfe (~115 kts).
    """

    def test_decel_zone_is_at_least_3_nm(self):
        ribbon = plan_path(
            dep_lat=45.42, dep_lon=-91.77, dep_alt_ft=1100.0,
            dep_heading=10.0,
            dest_lat=47.40, dest_lon=-94.77, dest_alt_ft=1380.0,
            dest_rwy_heading=270.0,
            dest_threshold_lat=47.40, dest_threshold_lon=-94.77,
            cruise_alt_ft=5000.0,
            v_stall=77.0,
        )
        g = ribbon.geometry
        # Haversine-ish approximation in nm between decel_start and
        # descent_start.
        def _nm(lat1, lon1, lat2, lon2):
            R = 3440.065  # earth radius in nm
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlmb = math.radians(lon2 - lon1)
            a = (math.sin(dphi/2)**2 +
                 math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2)
            return 2 * R * math.asin(math.sqrt(a))

        decel_nm = _nm(g.decel_start_lat, g.decel_start_lon,
                       g.descent_start_lat, g.descent_start_lon)
        self.assertGreaterEqual(
            decel_nm, 3.0,
            f"DECEL zone is only {decel_nm:.1f}nm — not enough room to "
            f"bleed speed from Vcruise to Vfe at idle."
        )


if __name__ == "__main__":
    unittest.main()
