"""Emitter: Peregrine spec -> X-Plane .acf, authored 100% from our own values.

NO stock aircraft is loaded or overwritten. We write only the P-lines we author
(flight model from the spec) and let X-Plane initialise its internal defaults for
everything we don't emit.

X-Plane wing slots are role-hardwired: 0/1 = main wing L/R, 8/9 = h-stab L/R,
10 = v-stab (the tail's damping + elevator/rudder routing only activate on 8/9/10).
.acf units: lengths FEET, masses POUNDS. Frame: _part_x=lateral(right),
_part_y=vertical(up), _part_z=longitudinal(aft) — = our model frame. Datum = our
model origin (all parts + CG in one consistent frame).

`powered=False` emits a glider (no engine/prop) — used to isolate the airframe
from the propeller, which has computed blade geometry we can't yet fully author.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen.acf import ACF          # noqa: E402  (only for _fmt)
from acf_gen.peregrine import build  # noqa: E402

FT = 3.280839895
LB = 2.2046226218
NSTAT = 10
OUT = Path.home() / "X-Plane 12" / "Aircraft" / "Peregrine Gen" / "peregrine.acf"


def emit(out_path: Path, powered: bool = True) -> dict:
    s = build()
    cg = s.cg
    P: list[tuple[str, object]] = []
    def add(path, val): P.append((path, val))
    def arr(prefix, value, n=NSTAT):
        for i in range(n):
            add(f"{prefix}/{i}", value)
        add(f"{prefix}/count", n)

    # ---------------- global ----------------
    add("acf/_name", "PEREGRINE GEN" + ("" if powered else " (glider)"))
    add("acf/_descrip", s.description)
    add("acf/_author", s.author)
    add("acf/_tailnum", "PRG01")
    add("acf/_is_glider", 0 if powered else 1)
    add("acf/_m_empty", s.empty_mass * LB)   # 5 kg empty (no payload)
    add("acf/_m_max", s.total_mass * LB)     # 7 kg operating (empty + 2 kg payload)
    add("acf/_m_fuel_max_tot", 0.0)
    # CG from the mass model. Two X-Plane quirks handled here:
    #  (1) _cg_in_mac=0 -> the CG arms are ABSOLUTE feet from datum, not %MAC;
    #      without it X-Plane discards _cgZ entirely and defaults the CG to z=0.
    #  (2) X-Plane measures _cgZ from a datum ~0.116 m AFT of our part datum (found
    #      empirically: the level static sit, and the CG sitting forward of the NP,
    #      both land at _cgZ ~ -1.2 ft, i.e. our computed CG shifted forward by this).
    #      Applying CGZ_OFFSET puts the effective CG at the true part-frame position
    #      (forward of the neutral point => positive static margin / stable).
    CGZ_OFFSET = -0.116   # metres — X-Plane places the CG ~0.116 m aft of our part
                          # datum; shift _cgZ forward so the effective CG lands at the
                          # real part-frame position (forward of NP = stable + level sit).
    cgz = cg[2] + CGZ_OFFSET
    add("acf/_cg_in_mac", 0)
    add("acf/_cgZ", cgz * FT)
    add("acf/_cgY", cg[1] * FT)
    add("acf/_cgZ_fwd", (cgz - 0.03) * FT)
    add("acf/_cgZ_aft", (cgz + 0.03) * FT)
    add("acf/_cgX_lft", -0.01)
    add("acf/_cgX_rgt", 0.01)
    # Control surfaces at realistic chord fractions. X-Plane is full blade-element:
    # deflecting a surface geometrically bends the rear (crat..1.0) of each section as
    # added camber, fed into the same 2-D polar. The earlier flat 0.70 chord + a
    # tau pre-division on the deflections was over-powered and flutter-prone; use
    # textbook fractions and emit the spec deflection LIMITS directly in degrees.
    add("acf/_ailn1_cratR", 0.30); add("acf/_ailn1_cratT", 0.30)
    add("acf/_elev1_cratR", 0.40); add("acf/_elev1_cratT", 0.40)
    add("acf/_rudd1_cratR", 0.40); add("acf/_rudd1_cratT", 0.40)
    add("acf/_flap1_cratR", 0.30); add("acf/_flap1_cratT", 0.30)
    am = {x.role: float(x.control.deflection_limits[1]) for x in s.surfaces if x.control}
    add("acf/_ailn1_up", am["main_wing"]); add("acf/_ailn1_dn", am["main_wing"])  # +/-20 deg
    add("acf/_elev1_up", am["h_stab"]);    add("acf/_elev1_dn", am["h_stab"])     # +/-25 deg
    add("acf/_rudd1_lf", am["v_stab"]);    add("acf/_rudd1_rt", am["v_stab"])     # +/-25 deg
    add("acf/_Vne_kts", float(s.limits.vne_kts))
    add("acf/_Vno_kts", float(s.limits.vno_kts))
    add("acf/_num_engn", 1 if powered else 0)
    add("acf/_num_prop", 1 if powered else 0)
    add("acf/_num_batteries", 1 if powered else 0)
    add("acf/_num_buses", 1); add("acf/_num_xmsn", 1 if powered else 0)
    add("acf/_num_generators", 0); add("acf/_num_inverters", 0)
    # Rotational inertia (radius-of-gyration^2, m^2/kg) DERIVED from the mass model,
    # not declared. X-Plane _Jxx/_Jyy/_Jzz_unitmass = ROLL/PITCH/YAW; our tensor is
    # (Ixx=pitch, Iyy=yaw, Izz=roll) in the model frame, so map by PHYSICAL axis.
    # Roll is floored at 0.01 so a light-winged layout can't drive it to ~0 (that
    # near-zero roll inertia was what fed the original divergence).
    ixx, iyy, izz = s.inertia
    mtot = s.total_mass
    # Floors at REALISTIC distributed-mass radii of gyration (~0.24-0.28 m for a
    # 1.45 m-span / 7 kg airframe). Point masses lumped on the centerline UNDER-
    # estimate inertia (esp. roll), and X-Plane goes numerically unstable on a small,
    # light airframe when inertia is too low (false ground crashes). These floors are
    # both more physically accurate AND stabilising.
    add("acf/_Jxx_unitmass", max(izz / mtot, 0.06))   # roll  <- Izz (longitudinal)
    add("acf/_Jyy_unitmass", max(ixx / mtot, 0.06))   # pitch <- Ixx (lateral)
    add("acf/_Jzz_unitmass", max(iyy / mtot, 0.08))   # yaw   <- Iyy (vertical)

    # ---------------- wings -----------------
    sb = {x.role: x for x in s.surfaces}
    # FIX1: X-Plane wing slots are role-hardwired — h-stab MUST be 8/9, v-stab MUST
    # be 10 (slots 2-7 are lifting-panel slots; the tail's damping + elevator/rudder
    # routing only activate on 8/9/10). Wrong slots = no yaw damping = the -253 deg/s.
    # main-wing dihedral 3 deg (both halves same positive sign — side is handled by
    # _part_x/_is_right_mult) gives roll restoring moment / spiral stability.
    plan = [(sb["main_wing"], 0, -1, 3.0), (sb["main_wing"], 1, +1, 3.0),
            (sb["h_stab"], 8, -1, 0.0),    (sb["h_stab"], 9, +1, 0.0),
            (sb["v_stab"], 10, 0, 90.0)]
    for surf, slot, side, dihed in plan:
        root = surf.root_position
        w = f"_wing/{slot}"
        add(f"{w}/_Croot", surf.sections[0].chord * FT)
        add(f"{w}/_Ctip", surf.sections[-1].chord * FT)
        add(f"{w}/_semilen_SEG", surf.semi_span * FT)
        add(f"{w}/_semilen_JND", 0.0)
        add(f"{w}/_sweep_design", 0.0)
        add(f"{w}/_dihed_design", dihed)
        add(f"{w}/_part_x", (side * root[0]) * FT)
        # FIX2: per-panel side sign — X-Plane multiplies this into lateral force AND
        # differential aileron sense (NOT derived from part_x). Without it both
        # ailerons deflect the same way -> no clean roll + standing asymmetry.
        add(f"{w}/_is_right_mult", float(side) if side != 0 else 1.0)
        add(f"{w}/_part_y", root[1] * FT)
        add(f"{w}/_part_z", root[2] * FT)
        add(f"{w}/_afl_file_1", surf.sections[0].airfoil)
        add(f"{w}/_afl_file_2", surf.sections[0].airfoil)
        # wing AREA: element count + per-rib chord distribution. Without these the
        # wing computes to ZERO area -> NaN -> X-Plane crash (root cause of all crashes).
        croot_ft = surf.sections[0].chord * FT
        ctip_ft = surf.sections[-1].chord * FT
        add(f"{w}/_els", 10)
        add(f"{w}/_part_specs_eq", 1)
        for r in range(11):
            add(f"{w}/_chord_for_RIB/{r}", croot_ft + (ctip_ft - croot_ft) * r / 10.0)
            add(f"{w}/_c_rat_for_RIB/{r}", 1.0)
            add(f"{w}/_c_off_for_RIB/{r}", 0.0)
        for t in ("_chord_for_RIB", "_c_rat_for_RIB", "_c_off_for_RIB"):
            add(f"{w}/{t}/count", 11)
        add(f"{w}/_var_incid", 0)   # v1: FIXED stabilizers + control surfaces for
                                    # stability. All-moving (=1) removed all fixed
                                    # stability -> spiral + inverted; re-add later.
        add(f"{w}/_incid_full_up", float(surf.control.deflection_limits[1]) if surf.control else 0.0)
        arr(f"{w}/_incidence", surf.sections[0].incidence)
        drives = [d[0] for d in (surf.control.drives if surf.control else [])]
        if "roll" in drives: arr(f"{w}/_ailn1", 1)
        if "flap" in drives: arr(f"{w}/_flap1", 1)
        if "pitch" in drives: arr(f"{w}/_elev1", 1)
        if "yaw" in drives: arr(f"{w}/_rudd1", 1)

    # ---------------- propulsion (skipped for glider isolation) ------------
    if powered:
        p = s.propulsion[0]
        rpm = float(p.design_rpm or 7000.0)
        add("_engn/0/_type", "ELE")
        add("acf/_power_max_limit", p.max_power_kw)
        add("_engn/0/_power_max_limit", p.max_power_kw)   # per-engine shaft power (kW) — the field ELE consumes for thrust
        add("_engn/0/_thrust_max_limit", 0.0)             # electric is power-driven, not thrust-driven (matches stock ALIA)
        add("acf/_battery_watt_hr_max", p.battery_wh)     # battery charged -> bus powered, endurance modeled
        add("_engn/0/_part_x", p.position[0] * FT)
        add("_engn/0/_part_y", p.position[1] * FT)
        add("_engn/0/_part_z", p.position[2] * FT)
        add("_engn/0/_RSC_max_pwr_ENGN", rpm)
        add("_engn/0/_RSC_redline_ENGN", rpm)
        add("acf/_RSC_redline_ENGN", rpm)
        add("acf/_RSC_maxgreen_ENGN", rpm)
        add("acf/_RSC_mingreen_ENGN", rpm * 0.15)
        add("acf/_RSC_mingov_ENGN", rpm * 0.15)
        add("acf/_RSC_idlespeed_ENGN", 0.0)
        add("_prop/0/_num_blades", float(p.propeller.num_blades))
        add("_prop/0/_prop_type", 0)
        add("_prop/0/_des_rpm_prp", rpm)
        add("_prop/0/_des_kts_acf", 45.0)
        add("_prop/0/_des_Mach_limit", 1.0)
        add("_prop/0/_des_AOA_pwr", 1.0)
        add("_prop/0/_max_AOA_deg", 90.0)
        add("_prop/0/_rad_mod", 1.0)
        add("_prop/0/_prop_dir", -1.0)
        add("_prop/0/_prop_gear_rat", 1.0)
        blade_r = p.propeller.diameter / 2.0
        add("_blad/0/_semilen_SEG", blade_r * FT)
        add("_blad/0/_Croot", 0.030 * FT)
        add("_blad/0/_Ctip", 0.018 * FT)
        add("_blad/0/_afl_file_1", "Clark-Y (good propeller).afl")
        add("_blad/0/_afl_file_2", "Clark-Y (good propeller).afl")
        add("_blad/0/_els", 10)
        # blade AREA: per-rib chord table, exactly like the wings. Without it the
        # blade integrates to ZERO area -> zero thrust / NaN (same class as the
        # wing zero-area crash). Endpoints match the _Croot/_Ctip above.
        bcroot_ft = 0.030 * FT
        bctip_ft = 0.018 * FT
        for r in range(11):
            add(f"_blad/0/_chord_for_RIB/{r}", bcroot_ft + (bctip_ft - bcroot_ft) * r / 10.0)
            add(f"_blad/0/_c_rat_for_RIB/{r}", 1.0)
            add(f"_blad/0/_c_off_for_RIB/{r}", 0.0)
        for tname in ("_chord_for_RIB", "_c_rat_for_RIB", "_c_off_for_RIB"):
            add(f"_blad/0/{tname}/count", 11)
        for i in range(NSTAT):
            add(f"_blad/0/_incidence/{i}", 30.0 - 18.0 * i / (NSTAT - 1))
        add("_blad/0/_incidence/count", NSTAT)

    # ---------------- gear ------------------
    # Strut calibration scaled from the stock Aerolite 103 (lightest powered plane,
    # 125 kg) down to our 7 kg airframe. The earlier strut was massively underdamped
    # (_damp 6 vs the ~20 a light plane needs) AND was MISSING the force-curve spline
    # knots (_strut_s1/s2/t1/t2 + _strut_comp) -> on spawn the strut bounced and
    # flipped the plane instead of holding it. These make it sit and roll.
    W_LB = s.empty_mass * LB
    for slot, leg in enumerate(s.gear[:3]):
        g = f"_gear/{slot}"
        add(f"{g}/_gear_x", leg.position[0] * FT)
        # leg.position is the WHEEL CONTACT point, but X-Plane's _gear_y is the strut
        # ATTACH point and the wheel ends up at (_gear_y - _leg_len - _tire_radius).
        # So lift the attach by (leg_len + tire) to land the wheel at the contact —
        # otherwise the wheels sit ~0.19 m too low and the plane is twice as tall as
        # it should be (high CG -> tips over / crashes).
        add(f"{g}/_gear_y", (leg.position[1] + leg.leg_length + leg.tire_radius) * FT)
        add(f"{g}/_gear_z", leg.position[2] * FT)
        add(f"{g}/_gear_type", 2)
        add(f"{g}/_leg_len", leg.leg_length * FT)
        add(f"{g}/_tire_radius", leg.tire_radius * FT)
        add(f"{g}/_tire_swidth", leg.tire_radius * 0.6 * FT)  # tire width (0 -> NaN contact patch)
        add(f"{g}/_tire_psi", 20.0)
        # spring holds full aircraft weight at ~0.06 ft compression, small preload so
        # it's firm at rest; damper at ~the Aerolite's damp/stiffness ratio (kills bounce).
        # soft, long-travel, well-damped: a stiff strut catapulted the plane upward
        # ~1.9 m/s on spawn (X-Plane places it with the gear compressed); soft + damped
        # absorbs that without launching.
        add(f"{g}/_strut_max_wgt_def", 0.25)
        add(f"{g}/_strut_max_wgt_frc", 3.0 * W_LB)
        add(f"{g}/_strut_preload_def", 0.05)
        add(f"{g}/_strut_preload_frc", 0.3 * W_LB)
        add(f"{g}/_strut_comp", 0.0)
        add(f"{g}/_strut_s1", 0.0)            # force-curve spline knots (from stock light plane)
        add(f"{g}/_strut_s2", 0.015625)
        add(f"{g}/_strut_t1", 0.12890625)
        add(f"{g}/_strut_t2", 0.252929688)
        add(f"{g}/_damp", 45.0)
        add(f"{g}/_cyc_time", 0.1)
        add(f"{g}/_gear_can_retract", 0)
        add(f"{g}/_gear_castors", 0)
        add(f"{g}/_gear_renders_geo", 1)   # draw X-Plane's generic gear (we have no OBJ yet)
        add(f"{g}/_lonE", 13.0)               # gear ground-reaction params (from stock)
        add(f"{g}/_latE", 0.0)
        add(f"{g}/_axiE", 0.0)
        add(f"{g}/_gear_brakes", 0 if leg.steerable else 1)
        add(f"{g}/_steerdeg_hispeed", 8.0 if leg.steerable else 0.0)
        add(f"{g}/_steerdeg_lospeed", 15.0 if leg.steerable else 0.0)

    # ---------------- fuselage body ----------------
    # X-Plane draws the fuselage from a station x ring mesh: _body/0/_geo_xyz/s,r,axis
    # (axis 0=lateral, 1=vertical, 2=longitudinal). With NO body there is no fuselage
    # (the prop disc floats) AND X-Plane has no geometry to place/orient the plane on
    # the ground, which made it spawn inverted. We sweep the spec's body stations into
    # RD-point ellipses (width x height), centred on the fuselage datum, nose->tail.
    import math
    if s.bodies:
        st = s.bodies[0].stations
        NS, RD = 18, 18                       # stations, ring points (match stock _r_dim 18)
        xs = [b.x for b in st]
        def wh(z):                            # interpolate (width,height) at longitudinal z
            if z <= xs[0]: return st[0].width, st[0].height
            if z >= xs[-1]: return st[-1].width, st[-1].height
            for k in range(len(xs) - 1):
                if xs[k] <= z <= xs[k + 1]:
                    f = (z - xs[k]) / (xs[k + 1] - xs[k])
                    return (st[k].width + (st[k + 1].width - st[k].width) * f,
                            st[k].height + (st[k + 1].height - st[k].height) * f)
            return st[-1].width, st[-1].height
        add("_body/0/_descrip", "Fuselage")
        add("_body/0/_part_x", 0.0); add("_body/0/_part_y", 0.0); add("_body/0/_part_z", 0.0)
        add("_body/0/_part_specs_eq", 1)
        add("_body/0/_part_specs_invis", 0)   # 0 = visible (stock planes set 1 because they use an OBJ)
        add("_body/0/_part_cd", 0.20)
        add("_body/0/_part_area_rule", 1.0)
        add("_body/0/_engn_for_body", -1)
        add("_body/0/_gear_for_body", -1)
        add("_body/0/_s_dim", NS)
        add("_body/0/_r_dim", RD)
        z0, z1 = xs[0], xs[-1]
        maxr = 0.0
        for si in range(NS):
            zz = z0 + (z1 - z0) * si / (NS - 1)
            w, h = wh(zz)
            maxr = max(maxr, w / 2, h / 2)
            for r in range(RD):
                th = 2 * math.pi * r / (RD - 1)         # r=0 and r=RD-1 both at top (closed loop)
                add(f"_body/0/_geo_xyz/{si},{r},0", (w / 2) * math.sin(th) * FT)   # lateral
                add(f"_body/0/_geo_xyz/{si},{r},1", (h / 2) * math.cos(th) * FT)   # vertical
                add(f"_body/0/_geo_xyz/{si},{r},2", zz * FT)                       # longitudinal
        add("_body/0/_part_rad", maxr * FT)

    # ---------------- write -----------------
    lines = ["I", "1200 Version", "ACF", "", "PROPERTIES_BEGIN"]
    lines += [f"P {path} {ACF._fmt(val)}" for path, val in P]
    lines += ["PROPERTIES_END", "PANEL_2D_BEGIN", "PANEL_2D_END",
              "PANEL_3D_BEGIN", "PANEL_3D_END"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"out": str(out_path), "powered": powered, "n_props": len(P),
            "mass_lb": round(s.empty_mass * LB, 2)}


if __name__ == "__main__":
    import sys as _s
    powered = "--glider" not in _s.argv
    print(emit(OUT, powered=powered))
