"""Aircraft sizing / stability calculator — derive the numbers, don't guess them.

Computes, from the parametric spec geometry:
  * each lifting surface's aerodynamic centre (quarter-chord) and 3-D lift slope
  * the longitudinal NEUTRAL POINT (wing + downwash-corrected tail)
  * the static margin of the current CG, and the CG needed for a target margin
  * the weight distribution (battery position) that puts the CG there
  * stall speed, wing loading, and the vertical-tail volume (yaw stability)

Frame: model Z = longitudinal, +Z aft (nose at -Z), metres. g, rho at sea level.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen.peregrine import build  # noqa: E402

RHO = 1.225      # kg/m^3 sea level
G = 9.80665
CLMAX = 1.3      # ~NACA 2412 max


def surf_props(srf):
    """area (gross, both sides), span, AR, CLalpha (/rad), AC longitudinal (model Z)."""
    n = 2 if srf.symmetric else 1
    area = n * srf.area
    # tip lateral extent = root offset + semi span; gross span tip-to-tip
    tip_lat = abs(srf.root_position[0]) + srf.semi_span
    span = (2 * tip_lat) if srf.symmetric else srf.semi_span
    ar = span * span / area if area else 0.0
    cla = 2 * math.pi * ar / (ar + 2) if ar else 0.0     # finite-wing lift slope
    ac_z = srf.root_position[2] + 0.25 * srf.mac          # quarter-chord, model Z
    return dict(area=area, span=span, ar=ar, cla=cla, ac=ac_z, mac=srf.mac)


def report():
    s = build()
    by = {x.role: x for x in s.surfaces}
    w = surf_props(by["main_wing"])
    h = surf_props(by["h_stab"])
    v = surf_props(by["v_stab"])
    cg = s.cg
    M = s.empty_mass

    # downwash gradient at the tail, and tail effectiveness
    deda = 2 * w["cla"] / (math.pi * w["ar"])
    eta_h = 0.9 * (1 - deda)

    # neutral point: lift-weighted centroid of wing + (downwash-corrected) tail ACs
    num = w["cla"] * w["area"] * w["ac"] + h["cla"] * h["area"] * eta_h * h["ac"]
    den = w["cla"] * w["area"] + h["cla"] * h["area"] * eta_h
    x_np = num / den
    mac = w["mac"]
    sm_now = (x_np - cg[2]) / mac

    SM_T = 0.12
    cg_req = x_np - SM_T * mac

    # weight distribution: solve battery z to hit cg_req (others fixed)
    others = [m for m in s.masses if m.name != "battery"]
    bat = next(m for m in s.masses if m.name == "battery")
    sum_others = sum(m.mass * m.position[2] for m in others)
    bat_z_req = (M * cg_req - sum_others) / bat.mass

    # performance
    W = M * G
    S = w["area"]
    v_stall = math.sqrt(2 * W / (RHO * S * CLMAX))
    wing_load = M / S
    # vertical tail volume coefficient (yaw): Vv = Sv*lv / (Sw*b)
    lv = v["ac"] - cg[2]
    Vv = (v["area"] * lv) / (w["area"] * w["span"])
    lh = h["ac"] - cg[2]
    Vh = (h["area"] * lh) / (w["area"] * mac)

    P = print
    P("================ PEREGRINE SIZING ================")
    P(f"surface     area(m2)  span(m)  AR    CLa(/rad)  AC_z(m)")
    for nm, d in (("main_wing", w), ("h_stab", h), ("v_stab", v)):
        P(f"  {nm:9s} {d['area']:7.4f}  {d['span']:6.3f}  {d['ar']:4.1f}  {d['cla']:6.2f}    {d['ac']:+.3f}")
    P("")
    P(f"downwash deps/dalpha   : {deda:.2f}   (tail eff eta_h={eta_h:.2f})")
    P(f"NEUTRAL POINT  x_np    : {x_np:+.3f} m   ({(x_np-by['main_wing'].root_position[2])/mac*100:.0f}% MAC)")
    P(f"current CG     x_cg    : {cg[2]:+.3f} m   ({(cg[2]-by['main_wing'].root_position[2])/mac*100:.0f}% MAC)")
    P(f"static margin (now)    : {sm_now*100:+.1f}% MAC   "
      f"{'STABLE' if sm_now>0.05 else 'MARGINAL/UNSTABLE'}")
    P("")
    P(f"--- for target static margin {SM_T*100:.0f}% ---")
    P(f"required CG  x_cg      : {cg_req:+.3f} m   ({(cg_req-by['main_wing'].root_position[2])/mac*100:.0f}% MAC)")
    P(f"=> battery z should be : {bat_z_req:+.3f} m   (currently {bat.position[2]:+.3f} m; "
      f"move {(bat_z_req-bat.position[2])*1000:+.0f} mm)")
    nose_z = -0.55
    P(f"   (nose is at z={nose_z:+.2f} m -> battery {'FITS' if bat_z_req>nose_z else 'is AHEAD OF THE NOSE - need lighter tail / heavier nose / move wing aft'})")
    P("")
    P(f"tail volumes  Vh={Vh:.2f} (pitch, want ~0.5)   Vv={Vv:.3f} (yaw, want ~0.02-0.05)")
    P(f"wing loading           : {wing_load:.1f} kg/m2")
    P(f"stall speed (CLmax {CLMAX}): {v_stall:.1f} m/s = {v_stall*1.944:.0f} kt")
    return dict(x_np=x_np, cg_req=cg_req, bat_z_req=bat_z_req, mac=mac, Vv=Vv, Vh=Vh)


if __name__ == "__main__":
    report()
