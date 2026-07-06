"""High-confidence aerodynamic sanity check for the Peregrine, via AeroSandbox.

Reads the SAME parametric spec that feeds the X-Plane .acf (acf_gen/peregrine.py)
and runs a vortex-lattice / aero-buildup analysis to answer: will this thing fly,
and fly nicely? Unlike a game flight model, this uses NeuralFoil airfoil polars at
the REAL Reynolds number (~200k) of a small UAV wing.

Checks:
  1. Low-Re airfoil reality (NACA 2412 wing, NACA 0009 tail) at Re~200k
  2. Whole-aircraft CL/CD/Cm vs alpha, max L/D
  3. Neutral point + static margin (the make-or-break stability number)
  4. Trim for level flight at cruise (alpha + does it balance)
  5. Static stability derivatives: pitch (Cma), yaw (Cnb), roll (Clb)
"""
import re
import sys
from pathlib import Path

import numpy as np
import aerosandbox as asb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen.peregrine import build  # noqa: E402

RHO = 1.225
G = 9.80665
WING_DIHEDRAL_DEG = 3.0   # applied to the main wing in emit.py


def naca(afl: str) -> str:
    m = re.search(r"NACA\s*(\d{4,5})", afl)
    return ("naca" + m.group(1)) if m else "naca0012"


def to_asb_wing(surf):
    """Our model frame (x=right, y=up, z=aft) -> ASB (x=aft, y=right, z=up)."""
    rx, ry, rz = surf.root_position
    vertical = (surf.role == "v_stab")
    dih = np.radians(0.0 if vertical else (WING_DIHEDRAL_DEG if surf.role == "main_wing" else 0.0))
    af = asb.Airfoil(naca(surf.sections[0].airfoil))
    xsecs = []
    for sec in surf.sections:
        if vertical:            # fin: span runs upward (+model y)
            lat, vert = rx, ry + sec.y
        else:                   # wing/h-stab: span runs outward (+model x), with dihedral
            lat = rx + sec.y * np.cos(dih)
            vert = ry + sec.y * np.sin(dih)
        lon = rz                # no sweep
        xsecs.append(asb.WingXSec(
            xyz_le=[lon, lat, vert], chord=sec.chord,
            twist=sec.incidence, airfoil=af,
        ))
    return asb.Wing(name=surf.role, xsecs=xsecs, symmetric=surf.symmetric)


def to_asb_fuse(body):
    xsecs = []
    for st in body.stations:
        xsecs.append(asb.FuselageXSec(
            xyz_c=[st.x, 0, 0],
            width=max(st.width, 1e-3), height=max(st.height, 1e-3),
        ))
    return asb.Fuselage(name=body.name, xsecs=xsecs)


def main():
    s = build()
    by = {x.role: x for x in s.surfaces}
    w = by["main_wing"]
    cg = s.cg                       # operating CG (model frame)
    M = s.total_mass
    W = M * G
    Sref = 2 * w.area              # gross wing area
    bref = 2 * (abs(w.root_position[0]) + w.semi_span)
    cref = w.mac
    cg_asb = [cg[2], cg[0], cg[1]]  # ASB ref = (aft, right, up)

    print("=" * 64)
    print("PEREGRINE AERO SANITY CHECK  (AeroSandbox + NeuralFoil)")
    print("=" * 64)
    print(f"mass(operating)={M:.2f} kg  W={W:.1f} N   Sref={Sref:.3f} m^2  "
          f"b={bref:.2f} m  MAC={cref:.3f} m")
    print(f"CG (model x,y,z)=({cg[0]:.3f},{cg[1]:.3f},{cg[2]:.3f}) m")

    # ---- 1. low-Re airfoil reality ----
    print("\n--- 1. AIRFOIL at low Reynolds (cruise 25 m/s) ---")
    for role, afl in (("wing", w.sections[0].airfoil),
                      ("tail", by["h_stab"].sections[0].airfoil)):
        chord = by["main_wing" if role == "wing" else "h_stab"].sections[0].chord
        Re = RHO * 25 * chord / 1.81e-5
        af = asb.Airfoil(naca(afl))
        al = np.linspace(-6, 16, 45)
        aero = af.get_aero_from_neuralfoil(alpha=al, Re=Re, mach=25 / 343)
        clmax = np.nanmax(aero["CL"]); astall = al[int(np.nanargmax(aero["CL"]))]
        cd0 = float(np.interp(0.0, al, aero["CD"]))
        print(f"  {role:4s} {naca(afl):9s} Re={Re/1e3:.0f}k: CLmax={clmax:.2f} @ "
              f"a={astall:.0f}deg   CD0~{cd0:.4f}")

    # ---- build airplane ----
    wings = [to_asb_wing(by["main_wing"]), to_asb_wing(by["h_stab"]), to_asb_wing(by["v_stab"])]
    fuses = [to_asb_fuse(s.bodies[0])] if s.bodies else []
    airplane = asb.Airplane(name="Peregrine", xyz_ref=cg_asb,
                            s_ref=Sref, b_ref=bref, c_ref=cref,
                            wings=wings, fuselages=fuses)

    # ---- 2. whole-aircraft polar ----
    print("\n--- 2. WHOLE-AIRCRAFT polar (V=25 m/s) ---")
    al = np.linspace(-4, 12, 17)
    op = asb.OperatingPoint(velocity=25, alpha=al)
    aero = asb.AeroBuildup(airplane=airplane, op_point=op).run()
    CL, CD, Cm = aero["CL"], aero["CD"], aero["Cm"]
    LD = CL / CD
    imax = int(np.nanargmax(LD))
    print(f"  CL range {CL.min():+.2f}..{CL.max():+.2f}   max L/D={LD[imax]:.1f} @ a={al[imax]:.0f}deg, CL={CL[imax]:.2f}")
    CL_cruise = W / (0.5 * RHO * 25**2 * Sref)
    print(f"  CL needed for level flight @25 m/s = {CL_cruise:.2f} "
          f"({'OK, below CLmax' if CL_cruise < CL.max() else 'TOO HIGH'})")

    # ---- 3 + 5. stability derivatives, NP, static margin ----
    print("\n--- 3/5. STABILITY (V=25 m/s, trimmed-ish alpha) ---")
    a_trim_guess = float(np.interp(CL_cruise, CL, al))
    op1 = asb.OperatingPoint(velocity=25, alpha=a_trim_guess, beta=0)
    der = asb.AeroBuildup(airplane=airplane, op_point=op1).run_with_stability_derivatives()
    Cma = float(der["Cma"]); CLa = float(der["CLa"])
    Cnb = float(der["Cnb"]); Clb = float(der["Clb"])
    SM = -Cma / CLa
    x_np = cg[2] + SM * cref       # NP aft of CG by SM*MAC (+z = aft)
    print(f"  Cm_alpha={Cma:+.3f}/rad  CL_alpha={CLa:.3f}/rad")
    print(f"  STATIC MARGIN = {SM*100:+.1f}% MAC   -> {'STABLE (pitch)' if SM>0.05 else 'MARGINAL/UNSTABLE'}")
    print(f"  neutral point z={x_np:+.3f} m   (CG z={cg[2]:+.3f} m)")
    print(f"  Cn_beta={Cnb:+.4f}/rad -> {'yaw STABLE' if Cnb>0 else 'yaw UNSTABLE'} (weathercock)")
    print(f"  Cl_beta={Clb:+.4f}/rad -> {'roll STABLE' if Clb<0 else 'roll UNSTABLE'} (dihedral effect)")

    # ---- 4. trim for level flight ----
    print("\n--- 4. TRIM for level flight ---")
    op2 = asb.OperatingPoint(velocity=25, alpha=al)
    aero2 = asb.AeroBuildup(airplane=airplane, op_point=op2).run()
    Cm2 = aero2["Cm"]
    sign = np.sign(Cm2)
    cross = np.where(np.diff(sign) != 0)[0]
    if len(cross):
        i = cross[0]
        a_tr = al[i] - Cm2[i] * (al[i+1]-al[i]) / (Cm2[i+1]-Cm2[i])
        CL_tr = float(np.interp(a_tr, al, aero2["CL"]))
        v_tr = float(np.sqrt(W / (0.5 * RHO * Sref * CL_tr))) if CL_tr > 0 else float("nan")
        print(f"  trims hands-off at a={a_tr:.1f} deg, CL={CL_tr:.2f} -> level-flight speed {v_tr:.1f} m/s ({v_tr*1.944:.0f} kt)")
    else:
        print(f"  no natural Cm=0 crossing in {al.min():.0f}..{al.max():.0f} deg "
              f"(Cm {Cm2.min():+.2f}..{Cm2.max():+.2f}); needs elevator/incidence trim")

    print("\n" + "=" * 64)
    print("VERDICT: pitch %s, yaw %s, roll %s, lift %s" % (
        "OK" if SM > 0.05 else "CHECK",
        "OK" if Cnb > 0 else "CHECK",
        "OK" if Clb < 0 else "CHECK",
        "OK" if CL_cruise < CL.max() else "CHECK"))
    print("=" * 64)


if __name__ == "__main__":
    main()
