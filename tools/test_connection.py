"""
Quick connection test for X-Plane 12 via native UDP (port 49000).
Run with: python tools/test_connection.py
X-Plane must be running with a flight loaded.
"""

import math
import socket
import struct
import time

XPLANE_IP = "127.0.0.1"
XPLANE_PORT = 49000
LOCAL_PORT = 49005
TIMEOUT = 3.0

DATAREFS = {
    0: "sim/flightmodel/position/indicated_airspeed",
    1: "sim/flightmodel/misc/h_ind",
    2: "sim/flightmodel/position/theta",
    3: "sim/flightmodel/position/phi",
    4: "sim/flightmodel/position/psi",
    5: "sim/flightmodel/position/latitude",
    6: "sim/flightmodel/position/longitude",
}

LABELS = {
    0: "Airspeed (kts)",
    1: "Altitude (ft)",
    2: "Pitch (deg)",
    3: "Roll (deg)",
    4: "Heading (deg)",
    5: "Latitude",
    6: "Longitude",
}


def subscribe(sock, freq_hz, index, dataref):
    header = b"RREF\0"
    payload = struct.pack("<5sii", header, freq_hz, index)
    dr = dataref.encode("ascii") + b"\0"
    dr = dr.ljust(400, b"\0")
    sock.sendto(payload + dr, (XPLANE_IP, XPLANE_PORT))


def unsubscribe_all(sock):
    for idx, dataref in DATAREFS.items():
        header = b"RREF\0"
        payload = struct.pack("<5sii", header, 0, idx)
        dr = dataref.encode("ascii") + b"\0"
        dr = dr.ljust(400, b"\0")
        sock.sendto(payload + dr, (XPLANE_IP, XPLANE_PORT))


def main():
    print("=" * 50)
    print("  UAV Project — X-Plane 12 Connection Test")
    print("=" * 50)
    print(f"\nConnecting to X-Plane at {XPLANE_IP}:{XPLANE_PORT} ...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", LOCAL_PORT))
    except OSError:
        print(f"  Port {LOCAL_PORT} in use, trying auto-assign...")
        sock.bind(("", 0))
    sock.settimeout(TIMEOUT)

    # Subscribe to all datarefs at 10 Hz
    for idx, dataref in DATAREFS.items():
        subscribe(sock, 10, idx, dataref)

    print(f"  Subscribed to {len(DATAREFS)} DataRefs at 10 Hz")
    print(f"  Waiting up to {TIMEOUT}s for data...\n")

    values = {}
    deadline = time.time() + TIMEOUT

    while time.time() < deadline:
        try:
            data, _ = sock.recvfrom(1024)
        except socket.timeout:
            break

        if not data.startswith(b"RREF"):
            continue

        payload = data[5:]
        for i in range(0, len(payload), 8):
            if i + 8 > len(payload):
                break
            index, value = struct.unpack_from("<if", payload, i)
            values[index] = value

        if len(values) >= len(DATAREFS):
            break  # got everything

    unsubscribe_all(sock)
    sock.close()

    if not values:
        print("❌  No data received from X-Plane.")
        print("\n  Checklist:")
        print("  1. Is X-Plane 12 running with a flight loaded (not the main menu)?")
        print("  2. Settings → Net Connections → Data → enable UDP")
        print(f"  3. Port should be {XPLANE_PORT} (default)")
        return

    print("✅  Data received!\n")
    print(f"  {'DataRef':<20} {'Value':>12}")
    print("  " + "-" * 34)
    for idx in sorted(DATAREFS.keys()):
        label = LABELS[idx]
        val = values.get(idx, math.nan)
        print(f"  {label:<20} {val:>12.4f}")

    print("\n  Connection is healthy — ready to train! 🚀")


if __name__ == "__main__":
    main()
