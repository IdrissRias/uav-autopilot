#!/usr/bin/env python3
"""
Extract physics constants from flight CSV logs.

Writes a measurement pass over recent flight logs to produce per-phase
numbers for:
  - decel_rate_clean       kts/sec bled at idle, clean config, level (CRUISE/DECELERATE)
  - decel_rate_flaps       kts/sec bled at idle, flaps extended (DESCENT)
  - sink_rate_descent      ft/sec lost in DESCENT phase
  - roll_response          actual bank / commanded roll (soldier obedience)
  - climb_rate             ft/sec gained in CLIMB phases

Use these to size ribbon phase distances instead of heuristics.
"""
from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Row:
    t: float
    mode: str
    spd: float
    alt: float
    pitch: float
    roll: float
    hdg: float
    thr: float
    roll_cmd: float
    pitch_cmd: float


def load(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            try:
                rows.append(Row(
                    t=float(d["time"]),
                    mode=d["mode"],
                    spd=float(d["telemetry_airspeed_kts"]),
                    alt=float(d["telemetry_altitude_ft"]),
                    pitch=float(d["telemetry_pitch_deg"]),
                    roll=float(d["telemetry_roll_deg"]),
                    hdg=float(d["telemetry_heading_deg"]),
                    thr=float(d["act_throttle"]),
                    roll_cmd=float(d["act_roll"]),
                    pitch_cmd=float(d["act_pitch"]),
                ))
            except (ValueError, KeyError):
                continue
    return rows


def linear_rate(rows: List[Row], attr: str) -> float | None:
    """Least-squares slope of attr wrt time. Returns None if <2 points or
    time span too small."""
    if len(rows) < 2:
        return None
    t0 = rows[0].t
    xs = [r.t - t0 for r in rows]
    ys = [getattr(r, attr) for r in rows]
    span = xs[-1] - xs[0]
    if span < 3.0:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


def find_segments(rows: List[Row], predicate) -> List[List[Row]]:
    """Contiguous runs where predicate(row) is True."""
    segs: List[List[Row]] = []
    cur: List[Row] = []
    for r in rows:
        if predicate(r):
            cur.append(r)
        else:
            if len(cur) >= 10:
                segs.append(cur)
            cur = []
    if len(cur) >= 10:
        segs.append(cur)
    return segs


def analyze_flight(path: str) -> Dict:
    rows = load(path)
    if not rows:
        return {"file": os.path.basename(path), "empty": True}

    out: Dict = {"file": os.path.basename(path), "n_rows": len(rows),
                 "duration_s": rows[-1].t - rows[0].t}

    # Mode histogram
    mode_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        mode_counts[r.mode] += 1
    out["modes"] = dict(mode_counts)

    # ── 1. Decel rate, clean config, idle throttle, level flight ──
    # We want the ACHIEVABLE rate (plane was actively bleeding), not the
    # average rate (which includes the equilibrium plateau where the PID
    # held throttle up and decel stalled at ~0 kts/sec). Filter to segments
    # where speed dropped at least 15 kts over the segment.
    def is_decel_clean(r: Row) -> bool:
        return (r.mode in ("CRUISE", "DECELERATE")
                and r.thr < 0.25
                and abs(r.pitch) < 5.0)  # level flight only
    segs = find_segments(rows, is_decel_clean)
    decel_rates = []
    for seg in segs:
        if seg[-1].spd - seg[0].spd > -15.0:
            continue  # not a real decel segment
        rate = linear_rate(seg, "spd")
        # Reject crash transients (rates faster than -3 kts/sec are not real)
        if rate is not None and -3.0 < rate < -0.1:
            decel_rates.append(rate)
    if decel_rates:
        out["decel_rate_clean_kps"] = {
            "n_segments": len(decel_rates),
            "median": sorted(decel_rates)[len(decel_rates)//2],
            "min": min(decel_rates),
            "max": max(decel_rates),
        }

    # ── 2. Descending flight with idle throttle (flaps likely extended) ──
    # Mode mapping collapses DESCENT→CRUISE in the CSV, so proxy it as:
    # CRUISE mode + altitude dropping + low throttle.
    def is_descending(r: Row) -> bool:
        return (r.mode == "CRUISE"
                and r.thr < 0.25
                and r.pitch < -3.0)  # nose-down = descending
    segs = find_segments(rows, is_descending)
    desc_spd_rates = []
    desc_sink_rates = []
    for seg in segs:
        if seg[0].alt - seg[-1].alt < 200:
            continue  # didn't actually descend
        sr = linear_rate(seg, "alt")   # ft/sec
        spr = linear_rate(seg, "spd")  # kts/sec
        if sr is not None and sr < -1.0:
            desc_sink_rates.append(sr)
        if spr is not None:
            desc_spd_rates.append(spr)
    if desc_sink_rates:
        out["descent_sink_rate_fps"] = {
            "n_segments": len(desc_sink_rates),
            "median": sorted(desc_sink_rates)[len(desc_sink_rates)//2],
            "min": min(desc_sink_rates),
            "max": max(desc_sink_rates),
        }
    if desc_spd_rates:
        out["descent_speed_rate_kps"] = {
            "n_segments": len(desc_spd_rates),
            "median": sorted(desc_spd_rates)[len(desc_spd_rates)//2],
            "min": min(desc_spd_rates),
            "max": max(desc_spd_rates),
        }

    # ── 4. Roll authority: bank angle achieved vs aileron saturation ──
    # During sustained aileron-saturated (|roll_cmd| > 0.7) intervals, how
    # steeply does the bank actually build? bank_rate_deg_per_sec at full
    # authority tells us how long the soldier needs to execute a turn.
    bank_rates = []
    # Find 2s windows of saturated aileron in same direction
    window = 30  # ~2 sec at 15 Hz
    for i in range(len(rows) - window):
        if all(abs(r.roll_cmd) > 0.7 and
               math.copysign(1, r.roll_cmd) == math.copysign(1, rows[i].roll_cmd)
               for r in rows[i:i+window]):
            dt = rows[i+window].t - rows[i].t
            if dt < 0.5:
                continue
            dbank = rows[i+window].roll - rows[i].roll
            if rows[i].roll_cmd < 0:
                dbank = -dbank  # normalize so faster+ = bank building
            rate = dbank / dt
            if 0.5 < rate < 30.0:  # plausible range
                bank_rates.append(rate)
    if bank_rates:
        bank_rates.sort()
        out["bank_rate_saturated_dps"] = {
            "n_samples": len(bank_rates),
            "median": bank_rates[len(bank_rates)//2],
        }

    # ── 5. Climb rate, CLIMB mode, high throttle ──
    def is_climbing(r: Row) -> bool:
        return r.mode == "CLIMB" and r.thr > 0.85
    segs = find_segments(rows, is_climbing)
    climb_rates = []
    for seg in segs:
        rate = linear_rate(seg, "alt")  # ft/sec
        if rate is not None and rate > 1.0:
            climb_rates.append(rate)
    if climb_rates:
        out["climb_rate_fps"] = {
            "n_segments": len(climb_rates),
            "median": sorted(climb_rates)[len(climb_rates)//2],
            "min": min(climb_rates),
            "max": max(climb_rates),
        }

    # ── 6. Top airspeed seen (proxy for Vcruise achieved) ──
    out["max_airspeed_kts"] = max(r.spd for r in rows)

    return out


def aggregate(results: List[Dict]) -> Dict:
    """Combine per-flight medians into a single set of numbers."""
    agg: Dict = {}
    for key in ("decel_rate_clean_kps", "descent_speed_rate_kps",
                "descent_sink_rate_fps", "climb_rate_fps",
                "bank_rate_saturated_dps"):
        medians = [r[key]["median"] for r in results if key in r]
        if medians:
            medians.sort()
            agg[key] = {
                "n_flights": len(medians),
                "grand_median": medians[len(medians)//2],
                "min_median": min(medians),
                "max_median": max(medians),
            }
    return agg


def fmt(v, digits=2):
    if v is None: return "—"
    return f"{v:.{digits}f}"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--since", default="20260419",
                   help="Only include files with this date prefix or later")
    args = p.parse_args()

    files = sorted([
        os.path.join(args.logs_dir, f)
        for f in os.listdir(args.logs_dir)
        if f.startswith("run-") and f.endswith(".csv")
    ])
    files = [f for f in files if os.path.basename(f)[4:12] >= args.since]

    print(f"Analyzing {len(files)} flight logs since {args.since}\n")

    results = []
    for path in files:
        r = analyze_flight(path)
        results.append(r)

    # Per-flight table
    print(f"{'flight':<28}  {'dur':>5}  {'decel_clean':>12}  "
          f"{'desc_spd':>10}  {'sink':>8}  {'climb':>8}  {'bank_rate':>10}")
    print("-" * 100)
    for r in results:
        if r.get("empty"):
            print(f"{r['file']:<28}  EMPTY")
            continue
        dc = r.get("decel_rate_clean_kps", {}).get("median")
        ds = r.get("descent_speed_rate_kps", {}).get("median")
        sr = r.get("descent_sink_rate_fps", {}).get("median")
        cr = r.get("climb_rate_fps", {}).get("median")
        br = r.get("bank_rate_saturated_dps", {}).get("median")
        print(f"{r['file']:<28}  {r['duration_s']:>5.0f}s  "
              f"{fmt(dc)+' kps':>12}  {fmt(ds)+' kps':>10}  "
              f"{fmt(sr)+' fps':>8}  {fmt(cr)+' fps':>8}  "
              f"{fmt(br,1)+' d/s':>10}")

    # Aggregate
    agg = aggregate(results)
    print("\n" + "=" * 60)
    print("AGGREGATE (median across flights)")
    print("=" * 60)
    for k, v in agg.items():
        print(f"\n{k}:")
        for kk, vv in v.items():
            print(f"  {kk:<18} {fmt(vv, 3) if isinstance(vv, float) else vv}")


if __name__ == "__main__":
    main()
