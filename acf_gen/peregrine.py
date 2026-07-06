"""The real Peregrine v4 aircraft spec — built from the CAD measurements.

Geometry was measured off the Fusion model in Blender (peregrine_assembly_v4.fbx);
control allocation per the user's intent; mass is an estimated 7 kg budget with the
battery placed forward to balance the tail-mounted pusher (CG tunable later — sliding
the battery is the real-world trim knob).

Blender frame (measured): X=fore/aft (nose -X), Y=span, Z=up.
Model frame (X-Plane-ish): X=right/span, Y=up, Z=aft (nose -Z).
Conversion:  model(X,Y,Z) = (blender_Y, blender_Z, blender_X)   -> see M().
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen.model import (  # noqa: E402
    AircraftSpec, LiftingSurface, Section, ControlSurface, Body, BodyStation,
    MassComponent, Propulsion, Propeller, GearLeg, OperatingLimits,
)

WING = "NACA 2412 (popular).afl"     # ~12% (wing is ~9.8%; closest stock for v1)
TAIL = "NACA 0009 (symmetrical).afl" # symmetric stabs/fin


def M(bx, by, bz):
    """Blender (fore/aft, span, up) -> model (span, up, aft)."""
    return (by, bz, bx)


def build() -> AircraftSpec:
    surfaces = [
        # Main wing: full-flying. Root at fuselage side (bY 0.153), tip bY 0.88.
        LiftingSurface(
            "main_wing", M(-0.3475, 0.153, -0.009),
            [Section(0.0, 0.19, WING, 1.0), Section(0.727, 0.19, WING, 1.0)],
            symmetric=True,
            control=ControlSurface("all_moving", 1.0, (0.0, 1.0), (-20, 20),
                                   drives=[("roll", "differential"),
                                           ("flap", "symmetric")]),
        ),
        # Horizontal tail: all-moving stabilator (pitch).
        LiftingSurface(
            "h_stab", M(0.275, 0.038, -0.0125),
            # -1.5 deg LE-down incidence: a cambered main wing + a 0-deg tail trims
            # nose-heavy; the small downward tail incidence sets level-flight trim.
            [Section(0.0, 0.10, TAIL, -1.5), Section(0.312, 0.10, TAIL, -1.5)],
            symmetric=True,
            control=ControlSurface("all_moving", 1.0, (0.0, 1.0), (-25, 25),
                                   drives=[("pitch", "symmetric")]),
        ),
        # Vertical tail: all-moving (yaw). Enlarged 1.6x (chord 0.16->0.256,
        # span 0.162->0.259) — the original was directionally UNSTABLE (Cn_beta<0);
        # the fuselage out-yaws a small fin. 1.6x gives solid weathercock stability.
        LiftingSurface(
            "v_stab", M(0.26, 0.0, 0.0255),
            [Section(0.0, 0.256, TAIL), Section(0.259, 0.256, TAIL)],
            symmetric=False,
            control=ControlSurface("all_moving", 1.0, (0.0, 1.0), (-25, 25),
                                   drives=[("yaw", "direct")]),
        ),
    ]

    # Slim fixed-wing fuselage (was a 0.30 m-wide blimp). Max ~0.18 m wide / 0.17 m tall.
    bodies = [Body("fuselage", [
        BodyStation(-0.55, 0.04, 0.04), BodyStation(-0.30, 0.16, 0.16),
        BodyStation(0.00, 0.18, 0.17),  BodyStation(0.30, 0.12, 0.13),
        BodyStation(0.55, 0.05, 0.05),
    ])]

    propulsion = [Propulsion(
        "electric", M(0.502, 0.0, -0.0125), max_power_kw=2.5,
        propeller=Propeller(diameter=0.30, num_blades=2, direction=1),
        battery_wh=600.0, design_rpm=7000.0,
    )]

    # Short legs (0.09 m) so the strut attaches at the fuselage belly instead of
    # piercing up through the frame. Mains under the wing/belly, slight splay.
    gear = [
        GearLeg("nose",  M(-0.46, 0.0,  -0.1875), leg_length=0.09, tire_radius=0.028, steerable=True),
        GearLeg("main_L", M(-0.20, -0.13, -0.1875), leg_length=0.09, tire_radius=0.028),
        GearLeg("main_R", M(-0.20,  0.13, -0.1875), leg_length=0.09, tire_radius=0.028),
    ]

    # Estimated 7 kg budget. Battery far forward to offset the tail pusher.
    # Wing + gear masses are placed at their TRUE spanwise stations (not lumped on
    # the centerline) so the computed ROLL inertia is physical — lumping them at x=0
    # gave kx~0.03 m (near-zero roll inertia) which is what fed the divergence.
    masses = [
        # --- EMPTY weight = 5.0 kg (airframe + propulsion + battery, no payload) ---
        MassComponent("fuselage_shell", 0.7, M(0.00, 0.0, 0.0)),
        MassComponent("wing_L",         0.3, M(-0.30, -0.50, 0.0)),   # half-wing at span centroid
        MassComponent("wing_R",         0.3, M(-0.30, +0.50, 0.0)),
        MassComponent("tail_struct",    0.3, M(0.33, 0.0, 0.05)),
        MassComponent("gear_nose",      0.10, M(-0.46, 0.0,  -0.1875)),
        MassComponent("gear_L",         0.15, M(-0.20, -0.13, -0.1875)),
        MassComponent("gear_R",         0.15, M(-0.20, +0.13, -0.1875)),
        MassComponent("motor",          0.5, M(0.50, 0.0, 0.0)),
        MassComponent("battery",        2.5, M(-0.48, 0.0, -0.02)),   # CG-trim knob (forward)
        # --- PAYLOAD = 2.0 kg removable mission load ---
        MassComponent("payload",        2.0, M(-0.35, 0.0, 0.0), is_payload=True),
    ]

    limits = OperatingLimits(vne_kts=90, vno_kts=70, vs_kts=22, vfe_kts=45,
                             g_pos=6, g_neg=-3, ceiling_m=3000)

    return AircraftSpec(
        name="Peregrine 7kg", author="Idriss",
        description="Peregrine v4 — 5 all-moving surfaces, electric pusher",
        surfaces=surfaces, bodies=bodies, masses=masses,
        propulsion=propulsion, gear=gear, limits=limits,
    )


if __name__ == "__main__":
    s = build()
    w = s.main_wing
    cg = s.cg
    span = 2 * w.semi_span
    area = 2 * w.area
    mac = w.mac
    # CG fore/aft is model Z (= blender X). Wing AC ~ LE + 25% chord.
    cg_aft = cg[2]
    le_aft = w.root_position[2]
    wing_ac = le_aft + 0.25 * mac
    pct_mac = (cg_aft - le_aft) / mac * 100
    mains_aft = -0.21
    print(f"name        : {s.name}")
    print(f"mass        : {s.empty_mass:.2f} kg  ({s.empty_mass*2.2046:.2f} lb)")
    print(f"CG (X,Y,Z)  : ({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f}) m  [Z = fore/aft]")
    print(f"wing        : span {span:.2f} m, area {area:.3f} m2, MAC {mac:.3f} m, AR {span**2/area:.1f}")
    print(f"wing AC     : Z={wing_ac:.3f} m ; CG at {pct_mac:.0f}% MAC")
    print(f"CG vs mains : CG Z={cg_aft:.3f}, mains Z={mains_aft:.3f} -> "
          f"{'FWD of mains OK' if cg_aft < mains_aft else 'AFT of mains - TIP RISK'} "
          f"({(mains_aft-cg_aft)*1000:.0f} mm fwd)")
    print(f"wing loading: {s.empty_mass/area:.1f} kg/m2")
    print(f"surfaces    : {[ (x.role, [d[0] for d in (x.control.drives if x.control else [])]) for x in s.surfaces ]}")
    print(f"prop        : pusher {s.propulsion[0].propeller.diameter} m, "
          f"{s.propulsion[0].max_power_kw} kW, {s.propulsion[0].battery_wh} Wh")
