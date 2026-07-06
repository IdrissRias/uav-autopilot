# .acf Danger Audit — what can make X-Plane spawn the plane broken

Rev 0 — 2026-06-21. Companion to `acf_gen/defaults.py`. Built by cataloguing
every property across 5 clean stock aircraft (`build_catalog.py`) and reading
the crash-capable subset field-by-field (`analyze_plumbing.py`).

## Method

- **52,150** literal property paths across the union of glider / ultralight /
  GA single / jet / eVTOL (every index expanded).
- Classified into **FLIGHT** (≈48k — geometry/mass/propulsion/gear/control,
  authored from the parametric design), **VISUAL** (≈730 — `.obj`/light links),
  and **PLUMBING** (≈2,990 — the inert pile this audit targets).
- Of the plumbing: **2,032 invariant** across all 5 refs (safe consensus) and
  **958 variable**. Of the variable, only **91** carry a state/behaviour risk
  tag — and on reading, almost all are engine / gear / control *model* fields
  (parametric), not inert plumbing.

## Key findings (the reassuring part)

1. **Failures are NOT in the `.acf`.** Every failure-arming field
   (`_*_fail_*`, `_jam`, `_inop`, `_warn_fire_EQ`, `_feather_on_engn_fail_EQ`)
   reads "no failure" in all references. X-Plane stores *armed* failures in
   runtime/`.prf` state, not the aircraft file. → No hidden "spawns failed"
   trap can ride along in a generated plane.

2. **Spawn energy is a parametric field, not a buried default.** Battery
   capacity is `acf/_battery_watt_hr_max` (glider 1000, eVTOL 65000 Wh); liquid
   fuel mass `acf/_m_fuel_max_tot` is `0` for electric. We set these in the
   propulsion spec. → "dead battery on spawn" is ours to control, not a trap.
   **VERIFY IN X-PLANE:** that the battery spawns *charged* (start-charge may be
   a separate `_batt_power_rat` / runtime value).

3. **Ground hold is benign.** `_no_default_chocks = 0` (chocks present → plane
   held, safe); no parking-brake-on-spawn trap.

4. **The danger is concentrated, not diffuse.** Crash-capable fields live in
   propulsion / gear / control / mass — all authored from the parametric design
   and correct by construction. The genuinely-inert remainder (199 annunciator
   slots, 51 custom-instrument slots, electrical buses, the ground "bouncer",
   autopilot toggles) contains **no "spawns broken" field**.

## Subsystem gating counts (set to UAV minimums in `defaults.py`)

| field | glider | eVTOL | UAV |
|---|---|---|---|
| `_num_engn` | 0 | 5 | **1** |
| `_num_prop` | 0 | 5 | **1** |
| `_num_batteries` | 1 | 5 | **1** |
| `_num_buses` | 1 | 1 | **1** |
| `_num_generators` | 0 | 0 | **0** |
| `_num_inverters` | 0 | 0 | **0** |
| `_num_xmsn` | 0 | 5 | **1** |
| `_num_cylinders` | 4 | 4 | **4** (ignored for electric) |

## Deliberate behaviour toggles (electric ops; none crash-capable)

| field | value | why |
|---|---|---|
| `_no_default_engine_fx` | 1 | electric — no piston/jet smoke + heat fx |
| `_rev_thrust_EQ` | 0 | no reverse thrust |
| `_park_non_vect_props_00_below_02` | 1 | park prop at low throttle (electric) |
| `_park_non_vect_props_90_below_05` | 1 | as above |

Everything else in the inert set stays at the glider's benign baseline.

## Open items — only an X-Plane load test can finally confirm

1. Battery spawns **charged** (not at 0%).
2. Setting a subsystem count to a minimum doesn't leave dangling slots X-Plane
   chokes on (vs. needing the full array emitted with benign values).
3. A from-scratch minimal `.acf` built on this default set actually **loads and
   flies** — the real correctness gate.
