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
        self._gear_state: bool | None = None  # track last commanded gear position
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

    def write_actuators(self, act: Actuators) -> None:
        self.set_throttle(act.throttle)
        self.set_roll(act.roll)
        self.set_pitch(act.pitch)
        self.set_yaw(act.yaw)
        self.set_brakes(act.brake_ratio)
        self.set_gear(act.gear_down)

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

    def _send_posi(self, lat: float, lon: float, alt_m: float, pitch: float, roll: float, heading: float) -> None:
        """Teleport aircraft via XPlaneConnect POSI packet."""
        # Format: header(5s) + ac_idx(b) + lat(d) + lon(f) + alt_m(f) + pitch(f) + roll(f) + hdg(f) + gear(f)
        data = struct.pack(b"<5sbdfffffff",
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
            self._send_posi(lat, lon, alt_m, 0.0, 0.0, heading)
            time.sleep(0.1)
        # Also send X-Plane reset commands as a fallback.
        for cmd in (
            "sim/operation/reset_flight",
            "sim/operation/reset_to_runway",
        ):
            self._send_cmnd(cmd)
            time.sleep(0.05)
        self._gear_state = None
        return True
