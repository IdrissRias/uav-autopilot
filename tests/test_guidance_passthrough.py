"""PassthroughGuidance must pass EVERY Targets field through.

The field-enumerating version silently dropped throttle_for_alt,
throttle_base, and vs_target_fpm — the controller flew classic mode
for a day while the engine believed the energy coupling was active
(crash 20260706_131530, ground impact at 225 kts in DESCENT).

This test iterates over the dataclass fields so ANY future field
added to Targets is automatically covered.
"""
from __future__ import annotations

import dataclasses
import unittest

from uav.core.guidance import PassthroughGuidance
from uav.sim.types import Targets, Telemetry


class TestPassthroughDropsNothing(unittest.TestCase):
    def test_every_field_survives(self):
        # Build a Targets with a distinct non-default value per field.
        sentinel_by_type = {
            "float | None": 0.777,
            "bool | None": True,
            "bool": True,
            "float": 0.777,
        }
        kwargs = {}
        for f in dataclasses.fields(Targets):
            kwargs[f.name] = sentinel_by_type.get(str(f.type), 0.777)
        desired = Targets(**kwargs)

        t = Telemetry(
            airspeed_kts=100.0, altitude_ft=2000.0, pitch_deg=0.0,
            roll_deg=0.0, heading_deg=90.0, timestamp=0.0,
        )
        out = PassthroughGuidance().compute(t, desired)

        for f in dataclasses.fields(Targets):
            self.assertEqual(
                getattr(out, f.name), getattr(desired, f.name),
                f"PassthroughGuidance dropped Targets.{f.name} — the "
                f"controller will never see it."
            )

    def test_none_setpoints_filled_from_telemetry(self):
        desired = Targets(heading_deg=None, altitude_ft=None,
                          airspeed_kts=None)
        t = Telemetry(
            airspeed_kts=123.0, altitude_ft=4321.0, pitch_deg=0.0,
            roll_deg=0.0, heading_deg=45.0, timestamp=0.0,
        )
        out = PassthroughGuidance().compute(t, desired)
        self.assertEqual(out.heading_deg, 45.0)
        self.assertEqual(out.altitude_ft, 4321.0)
        self.assertEqual(out.airspeed_kts, 123.0)


if __name__ == "__main__":
    unittest.main()
