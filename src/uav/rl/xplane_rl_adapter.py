"""X-Plane adapter for RL training.

Uses RREF index range 100-107 to avoid conflicts with the main autopilot
which uses 0-7. This allows training to run on a separate port without
index collisions.

Also adds a teleport_airborne() method that properly positions the aircraft
in the air with velocity.
"""
from __future__ import annotations

import math
import struct
import time

from uav.sim.xplane_udp import XPlaneUDP


class XPlaneRLAdapter(XPlaneUDP):
    """Extended adapter for RL training — conflict-free RREF indices + teleport."""

    # Use indices 100+ to avoid collision with autopilot (0-7)
    DATAREFS = {
        100: "sim/flightmodel/position/indicated_airspeed",
        101: "sim/flightmodel/misc/h_ind",
        102: "sim/flightmodel/position/theta",
        103: "sim/flightmodel/position/phi",
        104: "sim/flightmodel/position/psi",
        105: "sim/flightmodel/position/latitude",
        106: "sim/flightmodel/position/longitude",
        107: "sim/flightmodel/position/y_agl",
    }

    # Map our indices back to the standard telemetry fields
    _IDX_MAP = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 105: 5, 106: 6, 107: 7}

    def read_telemetry(self):
        """Override to map 100+ indices back to 0-7 for Telemetry construction."""
        from uav.sim.types import Telemetry

        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except Exception:
                break
            if not data.startswith(b"RREF"):
                continue
            payload = data[5:]
            for i in range(0, len(payload), 8):
                if i + 8 > len(payload):
                    break
                index, value = struct.unpack_from("<if", payload, i)
                # Map 100+ to 0-7 for internal storage
                mapped = self._IDX_MAP.get(index, index)
                self._last_values[mapped] = value
            self._last_ts = time.time()

        airspeed = self._last_values.get(0, math.nan)
        alt = self._last_values.get(1, math.nan)
        pitch = self._last_values.get(2, math.nan)
        roll = self._last_values.get(3, math.nan)
        heading = self._last_values.get(4, math.nan)
        lat = self._last_values.get(5, math.nan)
        lon = self._last_values.get(6, math.nan)
        agl_m = self._last_values.get(7, math.nan)
        return Telemetry(
            airspeed_kts=airspeed,
            altitude_ft=alt,
            pitch_deg=pitch,
            roll_deg=roll,
            heading_deg=heading,
            timestamp=self._last_ts,
            lat_deg=lat,
            lon_deg=lon,
            agl_m=agl_m,
        )

    def teleport_airborne(
        self,
        lat: float,
        lon: float,
        alt_ft: float,
        heading: float,
        speed_kts: float,
    ) -> None:
        """Teleport aircraft to a position in the air with velocity.

        Resets crash state, repairs aircraft, repositions, and sets velocity.
        """
        # Step 1: Fix the plane — reset crash state and repair all damage
        self._send_dref(0.0, "sim/operation/fix_all_systems")
        # Reset crash detection
        self._send_dref(0, "sim/flightmodel/engine/ENGN_running")
        # Repair all surfaces and systems
        self._send_dref(1.0, "sim/operation/fix_all_systems")
        time.sleep(0.1)

        # Step 2: Unpause if paused after crash
        self._send_dref(0.0, "sim/time/paused")

        # Step 3: Override position control
        self._send_dref(1.0, "sim/operation/override/override_planepath")
        time.sleep(0.1)

        # Step 4: Teleport via POSI packet
        alt_m = alt_ft * 0.3048
        self._send_posi(lat, lon, alt_m, -3.0, 0.0, heading)
        time.sleep(0.2)

        # Send POSI again for reliability
        self._send_posi(lat, lon, alt_m, -3.0, 0.0, heading)
        time.sleep(0.2)

        # Step 5: Set velocity (POSI doesn't set airspeed)
        speed_ms = speed_kts * 0.514444
        hdg_rad = math.radians(heading)
        vx = speed_ms * math.sin(hdg_rad)   # east component
        vy = 0.0                              # no vertical speed
        vz = -speed_ms * math.cos(hdg_rad)  # south component (negative = north)
        self._send_dref(vx, "sim/flightmodel/position/local_vx")
        self._send_dref(vy, "sim/flightmodel/position/local_vy")
        self._send_dref(vz, "sim/flightmodel/position/local_vz")

        # Step 6: Clean aircraft state
        self.set_gear(False)
        self.set_flaps(0.0)
        self.set_brakes(0.0)
        self.set_throttle(0.4)

        # Step 7: Release override — let physics take over
        time.sleep(0.2)
        self._send_dref(0.0, "sim/operation/override/override_planepath")
