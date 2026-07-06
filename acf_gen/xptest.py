"""Headless X-Plane test harness for the generated Peregrine .acf.

reload aircraft -> place airborne (wings level) -> unpause -> observe free
response -> judge STABLE vs DIVERGENT -> if stable, command each axis and
measure the rate response. Prints a structured verdict. All via UDP, no GUI.
"""
import socket, struct, time, sys

HOST, PORT = "127.0.0.1", 49000
DR = {0: "sim/flightmodel/position/phi", 1: "sim/flightmodel/position/theta",
      2: "sim/flightmodel/position/P", 3: "sim/flightmodel/position/Q",
      4: "sim/flightmodel/position/R", 5: "sim/flightmodel/position/alpha",
      6: "sim/flightmodel/position/beta", 7: "sim/flightmodel/position/indicated_airspeed",
      8: "sim/flightmodel/position/y_agl", 9: "sim/flightmodel/position/latitude",
      10: "sim/flightmodel/position/longitude", 11: "sim/flightmodel/position/elevation",
      12: "sim/time/paused"}
ROLL_D = "sim/cockpit2/controls/yoke_roll_ratio"
PITCH_D = "sim/cockpit2/controls/yoke_pitch_ratio"
YAW_D = "sim/cockpit2/controls/rudder_ratio"


def mk():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("", 0)); s.settimeout(0.05); return s
def cmnd(s, c): s.sendto(b"CMND\0" + c.encode("ascii") + b"\0", (HOST, PORT))
def dref(s, d, v): s.sendto(struct.pack("<5sf", b"DREF\0", float(v)) + d.encode("ascii").ljust(500, b"\0"), (HOST, PORT))
def posi(s, lat, lon, alt, pitch, roll, hdg):
    s.sendto(struct.pack("<5sbdddffff", b"POSI\0", 0, float(lat), float(lon), float(alt),
                         float(pitch), float(roll), float(hdg), 0.0), (HOST, PORT))
def subscribe(s):
    for i, d in DR.items():
        s.sendto(struct.pack("<5sii", b"RREF\0", 20, i) + d.encode("ascii").ljust(400, b"\0"), (HOST, PORT))


def pump(s, dur, hold=None):
    out = {i: [] for i in DR}; t0 = time.time()
    while time.time() - t0 < dur:
        if hold:
            for d, v in hold.items(): dref(s, d, v)
        try: data, _ = s.recvfrom(2048)
        except socket.timeout: continue
        if data.startswith(b"RREF"):
            p = data[5:]
            for o in range(0, len(p), 8):
                if o + 8 <= len(p):
                    idx, v = struct.unpack_from("<if", p, o)
                    if idx in out: out[idx].append(v)
    return out


def last(o, i): return o[i][-1] if o.get(i) else float("nan")
def rng(o, i): return (round(min(o[i]), 1), round(max(o[i]), 1)) if o.get(i) else (float("nan"),) * 2
def rms(o, i, n=50):
    v = o[i][-n:] if o.get(i) else []
    return (sum(x * x for x in v) / len(v)) ** 0.5 if v else float("nan")


def main():
    s = mk()
    print(">> reload_aircraft"); cmnd(s, "sim/operation/reload_aircraft")
    pump(s, 9)                        # wait for reload
    subscribe(s); pump(s, 1)
    # POSI placement is unsupported on this install (X-Plane logged 'Unknown
    # inet_msg_type "POSI"' -> no XPlaneConnect plugin). For a glider, reset_flight
    # tows it aloft, which gives the airspeed needed to judge stability.
    print(">> reset_flight (glider tow aloft)")
    cmnd(s, "sim/operation/reset_flight"); pump(s, 5)
    for _ in range(12):                       # robust unpause: dref + cmnd, verify
        dref(s, "sim/time/paused", 0); cmnd(s, "sim/operation/pause_off")
        o = pump(s, 0.4)
        if last(o, 12) == 0: break
    if last(o, 12) != 0:
        print(f"   paused={last(o,12)} -> X-Plane WILL NOT RUN (flight-model-error dialog/NaN). RESULT=BLOCKED_PAUSED")
        return
    print(">> observe free response 12s")
    o = pump(s, 12, hold={ROLL_D: 0, PITCH_D: 0, YAW_D: 0})
    repaused = (last(o, 12) == 1)
    Pr, Qr, Rr = rms(o, 2), rms(o, 3), rms(o, 4)
    ias = last(o, 7)
    stable = (not repaused) and Pr < 35 and Qr < 35 and Rr < 35 and ias > 3
    print(f"   roll{rng(o,0)} pitch{rng(o,1)}  rateRMS P={Pr:.1f} Q={Qr:.1f} R={Rr:.1f}  IAS={ias:.1f}kt  repaused={repaused}")
    print(f"   STABILITY: {'STABLE' if stable else 'DIVERGENT'}")
    if not stable:
        print("RESULT=FAIL_DIVERGENT")
        return
    print(">> control response")
    rp = pump(s, 2.5, hold={ROLL_D: 1.0, PITCH_D: 0, YAW_D: 0})
    rn = pump(s, 2.5, hold={ROLL_D: -1.0, PITCH_D: 0, YAW_D: 0})
    pu = pump(s, 2.5, hold={ROLL_D: 0, PITCH_D: 1.0, YAW_D: 0})
    yw = pump(s, 2.5, hold={ROLL_D: 0, PITCH_D: 0, YAW_D: 1.0})
    pump(s, 0.5, hold={ROLL_D: 0, PITCH_D: 0, YAW_D: 0})
    print(f"   roll+ -> P{rng(rp,2)}   roll- -> P{rng(rn,2)}   (expect opposite signs)")
    print(f"   pitch+ -> Q{rng(pu,3)}   yaw+ -> R{rng(yw,4)}")
    print("RESULT=STABLE_CHECK_CONTROLS")


if __name__ == "__main__":
    main()
