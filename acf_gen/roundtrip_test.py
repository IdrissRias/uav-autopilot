"""Proof that acf_gen.ACF is lossless.

Loads every stock .acf in the X-Plane install, re-renders it, and checks the
output is byte-identical to the original. Then demonstrates a surgical
mutation (change one property -> exactly one line differs). Read-only with
respect to the install: originals are never written.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen import ACF  # noqa: E402

XPLANE = Path.home() / "X-Plane 12"
AIRCRAFT = XPLANE / "Aircraft"


def roundtrip_all() -> bool:
    files = sorted(AIRCRAFT.rglob("*.acf"))
    if not files:
        print(f"No .acf files found under {AIRCRAFT}")
        return False

    print(f"Round-tripping {len(files)} stock aircraft (byte-exact check)\n")
    width = max(len(f.parent.name) for f in files)
    all_ok = True
    for f in files:
        original = f.read_bytes()
        acf = ACF.load(f)
        produced = acf.render().encode("utf-8")
        ok = produced == original
        all_ok &= ok
        flag = "OK  " if ok else "FAIL"
        print(
            f"  [{flag}] {f.parent.name:<{width}}  "
            f"v{acf.version}  {len(acf):>5} props  {len(original):>9,} B"
        )
        if not ok:
            # Locate the first divergence to make a failure actionable.
            o = original.decode("utf-8", "replace").split("\n")
            p = produced.decode("utf-8", "replace").split("\n")
            for i, (a, b) in enumerate(zip(o, p)):
                if a != b:
                    print(f"        first diff at line {i}:")
                    print(f"          orig: {a!r}")
                    print(f"          ours: {b!r}")
                    break
    return all_ok


def mutation_demo() -> None:
    cessna = AIRCRAFT / "Laminar Research/Cessna 172 SP/Cessna_172SP.acf"
    if not cessna.exists():
        return
    print("\nMutation demo (in memory, original untouched):")
    acf = ACF.load(cessna)
    before = acf.render().split("\n")

    print(f"  read  acf/_name      = {acf['acf/_name']!r}")
    print(f"  read  acf/_m_empty   = {acf.get_float('acf/_m_empty')} kg-ish units")
    print(f"  read  wing0 root chord _wing/0/_Croot = {acf.get_float('_wing/0/_Croot')}")

    acf.set("acf/_name", "Peregrine Test")
    acf.set("_wing/0/_Croot", 0.42)

    after = acf.render().split("\n")
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    print(f"  changed acf/_name and _wing/0/_Croot -> {len(changed)} lines differ "
          f"(expected 2)")
    for i in changed:
        print(f"    line {i}: {before[i]!r}  ->  {after[i]!r}")


if __name__ == "__main__":
    ok = roundtrip_all()
    mutation_demo()
    print("\n" + ("ALL ROUND-TRIPS BYTE-IDENTICAL ✅" if ok else "ROUND-TRIP FAILURES ❌"))
    sys.exit(0 if ok else 1)
