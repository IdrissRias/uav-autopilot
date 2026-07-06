"""Clean airborne stability test — teleport to altitude, wings level, cruise speed,
neutral controls, and watch open-loop. Isolates the AIRFRAME from the gear and the
tow. If it holds roughly level it's stable; if it rolls toward inverted it's a real
roll problem (sign/damping), not a ground issue.
"""
import socket, struct, time

HOST, PORT = "127.0.0.1", 49000
DR = {0: "sim/flightmodel/position/phi", 1: "sim/flightmodel/position/theta",
      2: "sim/flightmodel/position/P", 3: "sim/flightmodel/position/Q",
      4: "sim/flightmodel/position/R", 7: "sim/flightmodel/position/indicated_airspeed",
      8: "sim/flightmodel/position/y_agl", 12: "sim/time/paused",
      9: "sim/flightmodel/position/local_y", 10: "sim/flightmodel/position/beta",
      11: "sim/flightmodel/position/alpha"}
THRO = "sim/cockpit2/engine/actuators/throttle_ratio_all"
ROLL_D = "sim/cockpit2/controls/yoke_roll_ratio"
PITCH_D = "sim/cockpit2/controls/yoke_pitch_ratio"
YAW_D = "sim/cockpit2/controls/rudder_ratio"


def mk():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("", 0)); s.settimeout(0.05); return s
def cmnd(s, c): s.sendto(b"CMND\0" + c.encode("ascii") + b"\0", (HOST, PORT))
def dref(s, d, v): s.sendto(struct.pack("<5sf", b"DREF\0", float(v)) + d.encode("ascii").ljust(500, b"\0"), (HOST, PORT))
def sub(s):
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
def rms(o, i, n=40):
    v = o[i][-n:] if o.get(i) else []
    return (sum(x * x for x in v) / len(v)) ** 0.5 if v else float("nan")


def teleport(s, ly):
    # wings-level identity quaternion, north-ish heading, 22 m/s forward, slight descent
    for q, v in (("q[0]", 1.0), ("q[1]", 0.0), ("q[2]", 0.0), ("q[3]", 0.0)):
        dref(s, "sim/flightmodel/position/" + q, v)
    dref(s, "sim/flightmodel/position/local_y", ly + 300.0)   # +300 m up
    dref(s, "sim/flightmodel/position/local_vx", 0.0)
    dref(s, "sim/flightmodel/position/local_vy", -1.0)
    dref(s, "sim/flightmodel/position/local_vz", -22.0)        # forward (north) ~22 m/s
    for r in ("Prad", "Qrad", "Rrad", "P", "Q", "R"):
        dref(s, "sim/flightmodel/position/" + r, 0.0)


def main():
    s = mk()
    print(">> reload + reset (get a valid flight)")
    cmnd(s, "sim/operation/reload_aircraft"); pump(s, 9)
    sub(s); pump(s, 1)
    cmnd(s, "sim/operation/reset_to_runway"); pump(s, 2)
    for _ in range(10):
        dref(s, "sim/time/paused", 0); cmnd(s, "sim/operation/pause_off")
        o = pump(s, 0.4)
        if last(o, 12) == 0: break
    if last(o, 12) != 0:
        print(f"   paused={last(o,12)} RESULT=BLOCKED_PAUSED"); return
    ly = last(o, 9)
    print(f">> teleport to {ly+300:.0f} m, wings level, 22 m/s, throttle 10%")
    # hold the teleport state ~1.5 s so it settles into level cruise
    o = pump(s, 1.6, hold={THRO: 0.10, ROLL_D: 0, PITCH_D: 0, YAW_D: 0,
                           "sim/flightmodel/position/local_vz": -22.0,
                           "sim/flightmodel/position/q[0]": 1.0,
                           "sim/flightmodel/position/q[1]": 0.0,
                           "sim/flightmodel/position/q[2]": 0.0,
                           "sim/flightmodel/position/q[3]": 0.0})
    # also force position up once more then release
    teleport(s, last(o, 9) - 300.0 if last(o, 9) == last(o, 9) else ly)
    pump(s, 0.4, hold={THRO: 0.10})
    print(">> RELEASE: neutral controls, observe 12 s open-loop")
    o = pump(s, 12, hold={THRO: 0.10, ROLL_D: 0, PITCH_D: 0, YAW_D: 0})
    repaused = (last(o, 12) == 1)
    Pr, Qr, Rr = rms(o, 2), rms(o, 3), rms(o, 4)
    print(f"   roll{rng(o,0)} pitch{rng(o,1)} beta{rng(o,10)} alpha{rng(o,11)}")
    print(f"   rateRMS P={Pr:.1f} Q={Qr:.1f} R={Rr:.1f}  IAS={last(o,7):.1f}kt y_agl={last(o,8):.0f}m repaused={repaused}")
    rollmax = max(abs(x) for x in o[0]) if o.get(0) else 999
    stable = (not repaused) and rollmax < 60 and Pr < 30 and Qr < 30 and Rr < 30 and last(o, 7) > 5
    print(f"   final roll={last(o,0):.1f} pitch={last(o,1):.1f}")
    print(f"   AIRFRAME: {'STABLE — holds level' if stable else ('ROLLS OVER (roll hit %.0f)' % rollmax)}")
    print("RESULT=" + ("AIRFRAME_STABLE" if stable else "AIRFRAME_ROLL_PROBLEM"))


if __name__ == "__main__":
    main()
