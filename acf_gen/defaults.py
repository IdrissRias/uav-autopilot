"""UAV plumbing default value set for from-scratch .acf generation.

SCOPE - the INERT plumbing only: annunciators, custom instruments, electrical
buses, the ground "bouncer", autopilot behaviour toggles, weather radar, misc
global flags. NOT geometry / mass / propulsion / gear / control - those are
authored by the generator from the parametric design and inherit nothing.

HOW THIS WAS BUILT (full audit in schema/DANGER_AUDIT.md):
  * Every .acf property was catalogued across 5 clean stock aircraft
    (glider, ultralight, GA single, jet, eVTOL) - acf_gen/build_catalog.py.
  * The crash-capable fields were read individually - acf_gen/analyze_plumbing.py.
    Findings: the .acf carries NO armed-failure state (failures are runtime/.prf
    state in X-Plane), and spawn energy lives in the parametric propulsion field
    `_battery_watt_hr_max` - so the inert plumbing holds no "spawns broken" trap.
  * Benign baseline values are taken from the GLIDER (most systems-stripped clean
    reference) and frozen here as OUR set, with the deliberate overrides below.

v0: nothing here comes from the broken peregrine.acf. Final correctness is
confirmed by loading a generated .acf in X-Plane and reading its log.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen import ACF  # noqa: E402
from acf_gen.build_catalog import classify  # noqa: E402

GLIDER = (Path.home() / "X-Plane 12" / "Aircraft" / "Laminar Research"
          / "Schleicher ASK 21" / "ASK21.acf")

# Subsystem sizes for a simple single-motor electric fixed-wing. These gate how
# many indexed slots matter; everything else stays at the benign baseline.
UAV_COUNTS = {
    "acf/_num_engn": 1,         # one electric motor
    "acf/_num_prop": 1,         # one propeller
    "acf/_num_cylinders": 4,    # ignored for electric engine type; kept at the
                                #   value working electric stock uses (Alia = 4)
    "acf/_num_batteries": 1,    # single battery (Alia uses 5 for its scale)
    "acf/_num_buses": 1,        # one electrical bus
    "acf/_num_generators": 0,   # pure battery, no generator (matches Alia)
    "acf/_num_inverters": 0,
    "acf/_num_xmsn": 1,         # one transmission / powerplant
}

# Behaviour toggles set deliberately for electric ops. Each was checked in the
# danger audit; none can crash the plane - they just match electric operation.
UAV_TOGGLES = {
    "acf/_no_default_engine_fx": 1,             # electric: no piston/jet smoke+heat fx
    "acf/_rev_thrust_EQ": 0,                     # no reverse thrust
    "acf/_park_non_vect_props_00_below_02": 1,   # park prop at low throttle (electric)
    "acf/_park_non_vect_props_90_below_05": 1,
}


def build_inert_defaults() -> dict[str, str]:
    """Return {path: value_string} for the inert plumbing baseline."""
    g = ACF.load(GLIDER)
    out = {p: g.get(p) for p in g.paths() if classify(p) == "PLUMBING"}
    for overrides in (UAV_COUNTS, UAV_TOGGLES):
        for k, v in overrides.items():
            out[k] = v if isinstance(v, str) else ACF._fmt(v)
    return out


if __name__ == "__main__":
    d = build_inert_defaults()
    out = Path(__file__).resolve().parent / "schema" / "uav_inert_defaults.txt"
    out.write_text("\n".join(f"{k}\t{v}" for k, v in sorted(d.items())) + "\n")
    print(f"inert plumbing default set: {len(d)} fields -> {out}")
    print("\ndeliberate overrides applied:")
    for k in list(UAV_COUNTS) + list(UAV_TOGGLES):
        print(f"  {k:<42} = {d[k]}")
