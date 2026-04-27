"""Quick script to teleport aircraft to KFAR RWY 31 threshold via POSI packet."""
import socket
import struct
import time

XPLANE_IP = "127.0.0.1"
XPLANE_PORT = 49000

# KFAR RWY 31 threshold coordinates
# KFAR elevation is ~902ft MSL = 274.9m. Use exact elevation so wheels touch ground.
LAT = 46.92040
LON = -96.81570
ALT_M = 275.0  # right at ground level
HEADING = 310.0
PITCH = 0.0
ROLL = 0.0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# First: send reset commands to stop any motion
for cmd_str in ("sim/operation/reset_flight",):
    cmd = cmd_str.encode("ascii")
    data = struct.pack(f"<4sx{len(cmd)}sx".encode(), b"CMND", cmd)
    sock.sendto(data, (XPLANE_IP, XPLANE_PORT))
    print(f"CMND sent: {cmd_str}")
    time.sleep(0.1)

# POSI packet: header(5s) + ac_idx(b) + lat(d) + lon(d) + alt(d) + pitch(f) + roll(f) + hdg(f) + gear(f)
posi = struct.pack(
    b"<5sbdddffff",
    b"POSI\0", 0,
    float(LAT), float(LON), float(ALT_M),
    float(PITCH), float(ROLL), float(HEADING),
    1.0,  # gear down
)

for i in range(3):
    sock.sendto(posi, (XPLANE_IP, XPLANE_PORT))
    print(f"POSI sent ({i+1}/3): lat={LAT}, lon={LON}, alt={ALT_M:.1f}m, hdg={HEADING}°")
    time.sleep(0.3)

sock.close()
print("Done — aircraft should be at KFAR RWY 31 threshold, on the ground")
