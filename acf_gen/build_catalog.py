"""Build a complete property catalog across clean stock aircraft.

Goal: account for EVERY .acf property so none of the ~1,800 "plumbing" fields is
unexamined. For each property path we record its value in each reference plane,
whether it is invariant across them, a coarse class (FLIGHT / VISUAL / PLUMBING),
and risk tags flagging fields whose wrong value can break/crash the sim.

Output:
  acf_gen/schema/property_catalog.csv   - one row per property path (full detail)
  printed summary                       - counts + the danger subset to scrutinize
"""

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen import ACF  # noqa: E402

BASE = Path.home() / "X-Plane 12" / "Aircraft" / "Laminar Research"

# A spread chosen for systems coverage, all clean stock:
#   glider      - most systems-stripped (no engine): the "minimal plumbing" donor
#   ultralight  - simplest piston + prop, very light
#   GA single   - complete conventional systems
#   modern jet  - electrical/hyd/fadec-heavy reference
#   eVTOL       - electric propulsion + battery fields done right
REFS = {
    "glider": "Schleicher ASK 21/ASK21.acf",
    "ultralight": "Aero-Works Aerolite 103/Aerolite_103.acf",
    "ga_single": "Cessna 172 SP/Cessna_172SP.acf",
    "jet": "Cirrus Vision SF50/CirrusSF50.acf",
    "electric": "BETA Technologies Alia-250/ALIA-250.acf",
}

# Namespaces authored from the parametric design (NOT part of the default set).
FLIGHT_NS = ("_wing", "_body", "_part", "_gear", "_engn", "_prop", "_blad",
             "_sbrk", "_slid", "_door", "_wpna", "_wpn", "_strut")
VISUAL_NS = ("_obja", "_lite", "_lite_equip")

# acf/ second-tokens that are flight-model, not plumbing.
ACF_FLIGHT_HINTS = ("_m_", "_cg", "_Jxx", "_Jyy", "_Jzz", "_V", "_size", "_lift",
                    "_drag", "_cl", "_cd", "_cm", "_diff", "_elev", "_ail", "_rud",
                    "_flap", "_roll", "_pitch", "_yaw", "_con", "_stall", "_Mmo",
                    "_Vmo", "_with", "_exp", "_has_", "_force", "_ref", "_trim",
                    "_stab", "_slung", "_num", "_gear", "_brake", "_fuel", "_tank",
                    "_batt", "_is_glider", "_warn_", "_vmca")

# Risk tags: substrings that mark a field as state/behaviour-critical.
RISK = {
    "MASS":    ("_m_empty", "_m_max", "_m_fuel", "_m_disp", "_cg", "_jxx", "_jyy", "_jzz"),
    "CONTROL": ("_ail", "_elv", "_rud", "_flap", "_spoi", "_sbrk", "_vct", "_con",
                "ratio", "deflect", "_trim", "_incid", "_var_incid", "_lock"),
    "ENGINE":  ("throt", "idle", "_start", "running", "_run", "rpm", "power",
                "thrust", "_fuel", "_tank", "batt", "fadec", "ignit", "fire", "_des_"),
    "GEAR":    ("_gear", "strut", "tire", "brak", "retract", "steer", "_leg"),
    "FLTMODEL":("_exp", "experiment", "_with_", "_force", "_over", "vmca", "stall",
                "_size", "_diff"),
    "FAILURE": ("fail", "_jam", "inop", "broke", "_dead"),
    "INIT":    ("_des_", "_start", "_default", "_init", "arm", "_park", "_on_"),
}


def classify(path: str) -> str:
    ns = path.split("/", 1)[0]
    if ns in VISUAL_NS:
        return "VISUAL"
    if ns in FLIGHT_NS:
        return "FLIGHT"
    if ns == "acf":
        rest = path[len("acf/"):]
        if any(rest.startswith(h) or h in rest for h in ACF_FLIGHT_HINTS):
            return "FLIGHT"
        return "PLUMBING"
    return "PLUMBING"


def risk_tags(path: str) -> list:
    low = path.lower()
    return [tag for tag, subs in RISK.items() if any(s in low for s in subs)]


def main() -> None:
    acfs = {}
    for key, rel in REFS.items():
        p = BASE / rel
        if not p.exists():
            print(f"MISSING reference: {p}")
            continue
        acfs[key] = ACF.load(p)
    keys = list(acfs.keys())
    print(f"Loaded references: {keys}\n")

    # Union of all property paths across references.
    all_paths = set()
    for a in acfs.values():
        all_paths.update(a.paths())

    rows = []
    for path in sorted(all_paths):
        vals = {k: acfs[k].get(path) for k in keys}
        present = [v for v in vals.values() if v is not None]
        distinct = set(present)
        rows.append({
            "path": path,
            "class": classify(path),
            "risk": "|".join(risk_tags(path)),
            "invariant": len(distinct) == 1 and len(present) == len(keys),
            "n_distinct": len(distinct),
            "n_present": len(present),
            **{f"val_{k}": (vals[k] if vals[k] is not None else "") for k in keys},
        })

    # Write full catalog.
    out_dir = Path(__file__).resolve().parent / "schema"
    out_dir.mkdir(exist_ok=True)
    out_csv = out_dir / "property_catalog.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    by_class = Counter(r["class"] for r in rows)
    plumbing = [r for r in rows if r["class"] == "PLUMBING"]
    plumb_invariant = [r for r in plumbing if r["invariant"]]
    plumb_variable = [r for r in plumbing if not r["invariant"]]
    danger = [r for r in plumbing if r["risk"]]

    print(f"TOTAL property paths: {len(rows)}")
    print(f"  by class: {dict(by_class)}")
    print(f"\nPLUMBING (the inert default set): {len(plumbing)}")
    print(f"  invariant across all refs (adopt value directly): {len(plumb_invariant)}")
    print(f"  variable (must choose a UAV default):             {len(plumb_variable)}")
    print(f"\n>>> DANGER subset = PLUMBING with risk tags: {len(danger)} fields <<<")
    print("    (these are the plumbing fields a wrong value could break the sim with)")
    dn_by_tag = Counter()
    for r in danger:
        for t in r["risk"].split("|"):
            dn_by_tag[t] += 1
    print(f"    by tag: {dict(dn_by_tag)}")

    # top-level namespace breakdown of plumbing
    plumb_ns = Counter()
    for r in plumbing:
        p = r["path"]
        ns = "acf/" + p[len("acf/"):].split("/")[0].lstrip("_")[:0] + \
             (p[len("acf/"):].split("/")[0] if p.startswith("acf/") else "")
        ns = p.split("/")[0] if not p.startswith("acf/") else "acf/" + p[len("acf/"):].split("/")[0]
        plumb_ns[ns] += 1
    print("\nPLUMBING by namespace (top 25):")
    for ns, c in plumb_ns.most_common(25):
        print(f"    {ns:<28} {c}")

    print(f"\nFull catalog written: {out_csv}")


if __name__ == "__main__":
    main()
