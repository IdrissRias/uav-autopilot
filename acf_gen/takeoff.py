"""Powered runway takeoff test (no tow) for the generated Peregrine .acf.

reload -> place on runway -> full throttle, brakes off -> roll -> rotate at Vr ->
climb with a dead-simple wing-leveler + gentle pitch hold -> judge whether it can
hold attitude in clean free flight. All via UDP, no GUI. This removes the glider
tow (a 7 kg UAV roped behind a 540 kg Super Cub) as a confound.
"""
import socket, struct, time

HOST, PORT = "127.0.0.1", 49000
DR = {0: "sim/flightmodel/position/phi", 1: "sim/flightmodel/position/theta",
      2: "sim/flightmodel/position/P", 3: "sim/flightmodel/position/Q",
      4: "sim/flightmodel/position/R", 7: "sim/flightmodel/position/indicated_airspeed",
      8: "sim/flightmodel/position/y_agl", 12: "sim/time/paused",
      13: "sim/flightmodel/position/psi"}
THRO = "sim/cockpit2/engine/actuators/throttle_ratio_all"
BRAKE = "sim/cockpit2/controls/parking_brake_ratio"
ROLL_D = "sim/cockpit2/controls/yoke_roll_ratio"
PITCH_D = "sim/cockpit2/controls/yoke_pitch_ratio"
YAW_D = "sim/cockpit2/controls/rudder_ratio"


def mk():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(("", 0)); s.settimeout(0.05); return s
def cmnd(s, c): s.sendto(b"CMND\0" + c.encode("ascii") + b"\0", (HOST, PORT))
def dref(s, d, v): s.sendto(struct.pack("<5sf", b"DREF\0", float(v)) + d.encode("ascii").ljust(500, b"\0"), (HOST, PORT))
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


def main():
    s = mk()
    print(">> reload_aircraft (powered)"); cmnd(s, "sim/operation/reload_aircraft"); pump(s, 9)
    subscribe(s); pump(s, 1)
    print(">> reset_to_runway"); cmnd(s, "sim/operation/reset_to_runway"); pump(s, 2)
    for _ in range(12):
        dref(s, "sim/time/paused", 0); cmnd(s, "sim/operation/pause_off")
        o = pump(s, 0.4)
        if last(o, 12) == 0: break
    if last(o, 12) != 0:
        print(f"   paused={last(o,12)} RESULT=BLOCKED_PAUSED"); return

    # ---- STEP 1: SIT CHECK — does it rest on its wheels, or topple? ----
    print(">> SIT CHECK: 5s, throttle 0, brakes on, controls neutral")
    sit = {THRO: 0, BRAKE: 1.0, ROLL_D: 0, PITCH_D: 0, YAW_D: 0}
    o = pump(s, 5, hold=sit)
    Pr = (sum(v*v for v in o[2][-30:]) / max(1, len(o[2][-30:]))) ** 0.5 if o.get(2) else 9e9
    Rr = (sum(v*v for v in o[4][-30:]) / max(1, len(o[4][-30:]))) ** 0.5 if o.get(4) else 9e9
    sits = abs(last(o, 0)) < 20 and abs(last(o, 1)) < 20 and Pr < 15 and Rr < 15 and last(o, 12) == 0
    print(f"   roll={last(o,0):.1f} pitch={last(o,1):.1f}  rateRMS P={Pr:.1f} R={Rr:.1f}  y_agl={last(o,8):.2f}m")
    print(f"   SIT: {'STABLE on wheels' if sits else 'TOPPLES / unstable on ground'}")
    if not sits:
        print("RESULT=SIT_FAIL"); return

    # ---- STEP 2: TAKEOFF ROLL at realistic throttle (cruise is ~6%) ----
    base = {THRO: 0.35, BRAKE: 0.0, ROLL_D: 0, PITCH_D: 0, YAW_D: 0}
    psi0 = last(pump(s, 0.5, hold=base), 13)
    print(f">> takeoff roll @ 35% throttle, brakes off (runway hdg ~{psi0:.0f})")
    o = {}; t0 = time.time(); vr = 22.0
    while time.time() - t0 < 14:
        o = pump(s, 0.5, hold=base)
        if last(o, 7) >= vr or last(o, 8) > 2: break
    print(f"   reached IAS={last(o,7):.1f}kt y_agl={last(o,8):.2f}m roll={last(o,0):.1f} after {time.time()-t0:.0f}s")

    # ---- STEP 3: ROTATE + CLIMB with a simple wing-leveler ----
    print(">> rotate (nose up) until airborne")
    h = dict(base); h[PITCH_D] = 0.30
    t0 = time.time()
    while time.time() - t0 < 6:
        o = pump(s, 0.4, hold=h)
        if last(o, 8) > 4: break
    print(f"   liftoff at IAS={last(o,7):.1f}kt y_agl={last(o,8):.1f}m pitch={last(o,1):.1f}")

    print(">> climb 14s, wing-leveler (roll cmd = -0.03*phi) + gentle pitch hold @ 30%")
    rolls = []; alts = []; ias = []
    t0 = time.time()
    while time.time() - t0 < 14:
        phi = last(o, 0)
        rcmd = max(-0.6, min(0.6, -0.03 * phi)) if phi == phi else 0
        h = {THRO: 0.30, BRAKE: 0.0, ROLL_D: rcmd, PITCH_D: 0.08, YAW_D: 0}
        o = pump(s, 0.5, hold=h)
        rolls.append(last(o, 0)); alts.append(last(o, 8)); ias.append(last(o, 7))
    repaused = (last(o, 12) == 1)
    phi_rng = (round(min(rolls), 1), round(max(rolls), 1))
    climbed = alts[-1] - alts[0] if alts else 0
    held = (not repaused) and max(abs(x) for x in rolls) < 60 and alts[-1] > 5
    print(f"   roll{phi_rng}  alt {alts[0]:.0f}->{alts[-1]:.0f}m (climb {climbed:+.0f})  IAS {min(ias):.0f}-{max(ias):.0f}kt  repaused={repaused}")
    print(f"   pitchrate Q{rng(o,3)} yawrate R{rng(o,4)}")
    print(f"   VERDICT: {'FLIES — holds attitude in free flight' if held else 'STILL UNCONTROLLED'}")
    print("RESULT=" + ("POWERED_FLIES" if held else "POWERED_FAIL"))


if __name__ == "__main__":
    main()
