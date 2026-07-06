"""Zero in on the plumbing fields that actually require a decision / can bite.

Two outputs that matter for a safe default set:
  1. SUBSYSTEM GATING COUNTS (acf/_num_*) - set these to a UAV minimum and most
     indexed plumbing slots stop mattering entirely.
  2. GLOBAL (non-indexed) PLUMBING fields whose value VARIES across reference
     planes - these are the conscious choices; invariant ones are safe consensus.

Full lists are written to acf_gen/schema/; the risk-tagged subset is printed.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acf_gen import ACF  # noqa: E402
from acf_gen.build_catalog import REFS, BASE, classify, risk_tags  # noqa: E402


def norm(p: str) -> str:
    return re.sub(r"/\d+", "/N", p)


def main() -> None:
    acfs = {k: ACF.load(BASE / rel) for k, rel in REFS.items() if (BASE / rel).exists()}
    keys = list(acfs)
    all_paths = set()
    for a in acfs.values():
        all_paths.update(a.paths())

    out_dir = Path(__file__).resolve().parent / "schema"
    out_dir.mkdir(exist_ok=True)

    def vline(p):
        return "  ".join(f"{k}={acfs[k].get(p)}" for k in keys)

    # 1) Gating counts -------------------------------------------------------
    counts = sorted(p for p in all_paths if p.startswith("acf/_num"))
    print(f"== SUBSYSTEM GATING COUNTS: {len(counts)} ==")
    for p in counts:
        print(f"  {p:<32} {vline(p)}")

    # 2) Global plumbing fields, split invariant vs variable -----------------
    g_invariant, g_variable = [], []
    for p in sorted(all_paths):
        if classify(p) != "PLUMBING":
            continue
        if norm(p) != p:
            continue  # indexed slot, governed by a count - handle in bulk later
        vals = [acfs[k].get(p) for k in keys]
        present = [v for v in vals if v is not None]
        if len(set(present)) == 1 and len(present) == len(keys):
            g_invariant.append(p)
        else:
            g_variable.append(p)

    (out_dir / "plumbing_global_invariant.txt").write_text(
        "\n".join(f"{p}\t{acfs[keys[0]].get(p)}" for p in g_invariant)
    )
    (out_dir / "plumbing_global_variable.txt").write_text(
        "\n".join(f"{p}\t{vline(p)}" for p in g_variable)
    )

    print(f"\n== GLOBAL PLUMBING (non-indexed) ==")
    print(f"  invariant across all 5 refs (adopt directly): {len(g_invariant)}")
    print(f"  variable (a conscious choice):                {len(g_variable)}")

    # The scary subset: variable globals that are state/behaviour-critical.
    risky = [(p, risk_tags(p)) for p in g_variable if risk_tags(p)]
    print(f"\n>>> VARIABLE GLOBAL PLUMBING WITH RISK TAGS: {len(risky)} (read these) <<<")
    for p, tags in risky:
        print(f"  [{'|'.join(tags)}] {p}")
        print(f"        {vline(p)}")

    print(f"\nFull lists in {out_dir}/plumbing_global_*.txt")


if __name__ == "__main__":
    main()
