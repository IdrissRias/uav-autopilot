from __future__ import annotations

import math
import socket
import struct
import time
from typing import Dict

from .adapter_base import SimAdapter
from .types import Telemetry, Actuators


class XPlaneUDP(SimAdapter):
    RREF_HEADER = b"RREF\0"
    DREF_HEADER = b"DREF\0"
    CMND_HEADER = b"CMND\0"

    DATAREFS: Dict[int, str] = {
        0: "sim/flightmodel/position/indicated_airspeed",
        1: "sim/flightmodel/misc/h_ind",
        2: "sim/flightmodel/position/theta",
        3: "sim/flightmodel/position/phi",
        4: "sim/flightmodel/position/psi",
        5: "sim/flightmodel/position/latitude",
        6: "sim/flightmodel/position/longitude",
        7: "sim/flightmodel/position/y_agl",  # metres above ground
        8: "sim/flightmodel/position/vh_ind_fpm",  # vertical speed (ft/min)
        9: "sim/flightmodel/position/groundspeed",  # m/s
        10: "sim/flightmodel/position/alpha",  # angle of attack (deg) — the
                                               # real envelope variable (stall is
                                               # an AoA event, not a speed event)
        11: "sim/flightmodel/position/Q",      # pitch RATE (deg/s), body axis —
                                               # true measured rate for the pitch
                                               # damper; finite-differencing pitch
                                               # is noisy+lagged and drove a PIO
    }

    def __init__(
        self,
        xplane_ip: str = "127.0.0.1",
        xplane_port: int = 49000,
        local_port: int = 49005,
        freq_hz: int = 10,
    ) -> None:
        self.xplane_addr = (xplane_ip, xplane_port)
        self.local_port = int(local_port)
        self.freq_hz = freq_hz
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # local_port=0 means "auto-pick a free port"
        bind_port = 0 if self.local_port == 0 else self.local_port
        try:
            self.sock.bind(("", bind_port))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to bind UDP port {self.local_port}. Address in use. "
                "Set ports.local_port=0 (auto) or pick a different port."
            ) from exc
        # if auto-picked, store the actual port
        self.local_port = int(self.sock.getsockname()[1])
        self.sock.settimeout(0.05)

        self._last_values: Dict[int, float] = {}
        self._last_ts: float = 0.0
        self._gear_state: bool | None = None   # track last commanded gear position
        self._flap_state: float | None = None  # track last commanded flap position
        self._home: tuple | None = None  # (lat, lon, alt_m, heading) set on first arm

        for idx, dataref in self.DATAREFS.items():
            self._send_rref(self.freq_hz, idx, dataref)

    def _send_rref(self, freq_hz: int, index: int, dataref: str) -> None:
        payload = struct.pack("<5sii", self.RREF_HEADER, freq_hz, index)
        dr = dataref.encode("ascii") + b"\0"
        dr = dr.ljust(400, b"\0")
        self.sock.sendto(payload + dr, self.xplane_addr)

    def _send_dref(self, value: float, dataref: str) -> None:
        payload = struct.pack("<5sf", self.DREF_HEADER, float(value))
        dr = dataref.encode("ascii") + b"\0"
        dr = dr.ljust(500, b"\0")
        self.sock.sendto(payload + dr, self.xplane_addr)

    def _send_cmnd(self, command: str) -> None:
        # Command trigger packet: CMND\0 + command + null
        msg = self.CMND_HEADER + command.encode("ascii") + b"\0"
        self.sock.sendto(msg, self.xplane_addr)

    def read_telemetry(self) -> Telemetry:
        # Re-subscribe if data is stale (X-Plane may have restarted)
        if self._last_ts and (time.time() - self._last_ts) > 5.0:
            for idx, dataref in self.DATAREFS.items():
                self._send_rref(self.freq_hz, idx, dataref)
            self._last_values.clear()  # don't serve ancient cached values
            self._last_ts = 0.0

        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                break

            if not data.startswith(b"RREF"):
                continue
            payload = data[5:]
            for i in range(0, len(payload), 8):
                if i + 8 > len(payload):
                    break
                index, value = struct.unpack_from("<if", payload, i)
                self._last_values[index] = value
            self._last_ts = time.time()

        airspeed = self._last_values.get(0, math.nan)
        alt = self._last_values.get(1, math.nan)
        pitch = self._last_values.get(2, math.nan)
        roll = self._last_values.get(3, math.nan)
        heading = self._last_values.get(4, math.nan)
        lat = self._last_values.get(5, math.nan)
        lon = self._last_values.get(6, math.nan)
        agl_m = self._last_values.get(7, math.nan)
        vs_fpm = self._last_values.get(8, math.nan)
        gs_ms = self._last_values.get(9, math.nan)
        gs_kts = gs_ms * 1.94384 if not math.isnan(gs_ms) else math.nan  # m/s → kts
        alpha_deg = self._last_values.get(10, math.nan)   # measured AoA
        pitch_rate = self._last_values.get(11, math.nan)  # measured pitch rate (deg/s)
        # Fallback: if X-Plane's alpha isn't flowing, compute it from geometry.
        # AoA = pitch - flight-path-angle, flight-path = asin(vertical / TAS).
        if math.isnan(alpha_deg) and not (math.isnan(pitch) or math.isnan(vs_fpm)
                                          or math.isnan(airspeed)):
            v_fps = airspeed * 1.68781
            if v_fps > 1.0:
                gamma = math.degrees(math.asin(
                    max(-1.0, min(1.0, (vs_fpm / 60.0) / v_fps))))
                alpha_deg = pitch - gamma
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
            vs_fpm=vs_fpm,
            groundspeed_kts=gs_kts,
            alpha_deg=alpha_deg,
            pitch_rate_deg_s=pitch_rate,
        )

    def write_actuators(self, act: Actuators) -> None:
        self.set_throttle(act.throttle)
        self.set_roll(act.roll)
        self.set_pitch(act.pitch)
        self.set_yaw(act.yaw)
        self.set_brakes(act.brake_ratio)
        self.set_gear(act.gear_down)
        self.set_flaps(act.flap_ratio)

    def set_throttle(self, value: float) -> None:
        self._send_dref(value, "sim/cockpit2/engine/actuators/throttle_ratio_all")

    def set_roll(self, value: float) -> None:
        self._send_dref(value, "sim/cockpit2/controls/yoke_roll_ratio")

    def set_pitch(self, value: float) -> None:
        self._send_dref(value, "sim/cockpit2/controls/yoke_pitch_ratio")

    def set_yaw(self, value: float) -> None:
        # Write both yoke heading and rudder ratio for broader aircraft support.
        self._send_dref(value, "sim/cockpit2/controls/yoke_heading_ratio")
        self._send_dref(value, "sim/cockpit2/controls/rudder_ratio")

    def set_brakes(self, value: float) -> None:
        # sim/flightmodel/controls/parkbrake is the writable parking brake (0.0=off, 1.0=on).
        # left/right_brake_ratio handle the wheel brakes directly.
        self._send_dref(value, "sim/flightmodel/controls/parkbrake")
        self._send_dref(value, "sim/cockpit2/controls/left_brake_ratio")
        self._send_dref(value, "sim/cockpit2/controls/right_brake_ratio")

    def set_flaps(self, ratio: float) -> None:
        """Set flap position using SF50 notch commands.
        SF50 has 2 notches: 0=up, 1=50%, 2=full.
        ratio 0.0 → notch 0, 0.01-0.74 → notch 1, 0.75-1.0 → notch 2.
        Re-sends commands every 60 ticks to ensure X-Plane received them."""
        if ratio < 0.01:
            target_notch = 0
        elif ratio < 0.75:
            target_notch = 1
        else:
            target_notch = 2

        current_notch = self._flap_state if self._flap_state is not None else 0
        current_notch = int(round(current_notch))

        # Count ticks at same target to periodically re-send
        if not hasattr(self, '_flap_ticks'):
            self._flap_ticks = 0
        self._flap_ticks += 1

        if target_notch == current_notch:
            # Re-send every 60 ticks (~3s) to ensure X-Plane got it
            if self._flap_ticks % 60 != 0:
                return

        self._flap_ticks = 0
        # Reset to notch 0 first, then step to target (absolute positioning)
        # This avoids drift from missed commands
        if current_notch != target_notch or self._flap_state is None:
            # Retract fully first
            for _ in range(3):
                self._send_cmnd("sim/flight_controls/flaps_up")
            # Then step to target
            for _ in range(target_notch):
                self._send_cmnd("sim/flight_controls/flaps_down")
            print(f"[FLAPS] notch {current_notch} → {target_notch} (ratio={ratio:.2f})")

        self._flap_state = float(target_notch)

    def set_gear(self, gear_down: bool) -> None:
        # sim/cockpit/switches/gear_handle_status is read-only — use commands instead.
        # Only send when state changes to avoid spamming X-Plane every loop tick.
        if gear_down == self._gear_state:
            return
        self._gear_state = gear_down
        if gear_down:
            self._send_cmnd("sim/flight_controls/landing_gear_down")
        else:
            self._send_cmnd("sim/flight_controls/landing_gear_up")

    def set_home(self, lat: float, lon: float, alt_m: float, heading: float) -> None:
        """Store the takeoff position so reset_flight can teleport back to it.
        Only set once — never overwrite after the first arm so crashes don't poison home."""
        if self._home is None:
            self._home = (lat, lon, alt_m, heading)
            print(f"HOME set: lat={lat:.5f} lon={lon:.5f} alt={alt_m:.1f}m hdg={heading:.1f}°")

    def _send_posi(self, lat: float, lon: float, alt_m: float, pitch: float, roll: float, heading: float) -> None:
        """Teleport aircraft via XPlaneConnect POSI packet."""
        # Format: header(5s) + ac_idx(b) + lat(d) + lon(d) + alt_m(d) + pitch(f) + roll(f) + hdg(f) + gear(f)
        data = struct.pack(b"<5sbdddffff",
            b"POSI\0", 0,
            float(lat), float(lon), float(alt_m),
            float(pitch), float(roll), float(heading),
            1.0,  # gear down for takeoff
        )
        self.sock.sendto(data, self.xplane_addr)

    def reset_flight(self) -> bool:
        # If we have a stored home position, teleport there first for a clean reset.
        if self._home:
            lat, lon, alt_m, heading = self._home
            print(f"RESET: teleporting to home lat={lat:.5f} lon={lon:.5f} alt={alt_m+3:.1f}m hdg={heading:.1f}°")
            # +3m safety margin so terrain mesh rounding never spawns us underground.
            self._send_posi(lat, lon, alt_m + 3.0, 0.0, 0.0, heading)
            time.sleep(0.1)
        else:
            print("RESET: no home stored — relying on X-Plane reset commands only")
        # Also send X-Plane reset commands as a fallback.
        for cmd in (
            "sim/operation/reset_flight",
            "sim/operation/reset_to_runway",
        ):
            self._send_cmnd(cmd)
            time.sleep(0.05)
        self._gear_state = None
        return True
