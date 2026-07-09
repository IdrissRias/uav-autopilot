# Peregrine Autopilot: How It Thinks

This document explains the autopilot from a logical perspective: what each
layer must achieve, how we reason about it, and which flight taught us each
rule. It deliberately avoids gain values and dataref names; the code is the
reference for those. Sections marked **[OPEN]** are questions awaiting a
ruling.

---

## 1. The Big Picture

The system is a chain of four roles, each blind to the layers below it:

```
RIBBON  ──►  ENGINE  ──►  CONTROLLER  ──►  ADAPTER
the plan     the commander  the soldier     the body
```

- **The Ribbon** (planner) draws the entire flight before takeoff: a
  polyline the plane can physically fly, plus a list of keyframes (stable
  command states) with triggers that advance from one to the next.
- **The Engine** walks the keyframes. Every tick it resolves the current
  keyframe into concrete orders: a heading, an altitude or sink rate, a
  speed, a configuration. It holds all situational judgment (when flaps are
  safe, when a landing is committed, when power is forbidden).
- **The Controller** is a dumb soldier. It receives orders and moves
  the stick, throttle, and rudder to satisfy them. It contains control
  theory, never policy.
- **The Adapter** turns normalized actuator commands into whatever the
  body understands. Today that body is X-Plane over UDP; for the real
  Peregrine airframe it becomes servo PWM plus a surface mixer. Nothing
  above this layer changes when we swap.

Why the split matters: every layer can be replaced or tested alone, and
when something goes wrong, the failure names its own layer. A wrong plan is
a ribbon bug; a wrong reaction is a controller bug; a wrong decision is an
engine bug.

**Lesson that proved the boundary matters:** a passthrough shim between
engine and controller silently dropped fields it did not know about. The
engine spent a day commanding an energy doctrine the controller never
heard. The fix was structural (copy everything, enumerate nothing), and the
regression test iterates the dataclass so any future field is covered
automatically.

---

## 2. The Ribbon

The ribbon is the single source of truth for geometry. Two principles:

**Everything is back-solved from the runway.** The landing chain is
computed threshold-first: touchdown point, flare start, final approach,
descent point, deceleration leg, join point. The cruise then aims at the
front door of that chain (the join), never at the airport itself. Arriving
"at the destination" is meaningless; arriving aligned, configured, and slow
at the start of the glidepath is the actual goal.

**Every curve is flyable by construction.** Turn radius comes from
physics (R = V² / g·tanφ at the bank limit). The departure is a tangent
arc that sweeps until it faces the join ("half a circle until it points at
the destination"); interior corners are fillet arcs the plane can hold at
fixed bank. The polyline the follower chases and the geometry the triggers
fire on are the same object.

Other ribbon decisions:

- **Altitude is bought by trip length.** Climbing costs time; higher
  cruise is faster. Below the break-even distance the climb never pays for
  itself, so short hops stay low and long trips ramp up.
- **The pattern inserts itself only when needed.** A base leg appears
  only when the single turn onto final would exceed what a fillet can
  honestly fly. Loops get tight patterns; the join run-up scales with trip
  length.
- **Trigger semantics: "arrived along the route."** A fix fires only when
  the follower's progress along the polyline has reached it. Being near a
  fix, or having it behind you, proves nothing: a loop's outbound leg
  physically overlaps the arrival corridor, and one flight cascaded through
  five phases in a single tick before this rule existed.
- **Runway axes come from endpoint coordinates, never stored headings.**
  Stored headings are the designator rounded to ten degrees; one runway was
  off by a full ten. That single error explains both "rolled into the
  grass" (steering faithfully held a rotated axis) and "landed beside the
  runway, parallel to it" (a flawless approach onto a phantom centerline).

**[OPEN]** Cruise speed by trip length (the same logic as altitude) is
designed but not built: plan fast when the trip affords it, since arcs and
deceleration legs are functions of the planned speed. Needs one
calibration flight to learn the true max-continuous cruise.

---

## 3. The Energy Doctrine

The core mental model, in one sentence: **altitude and speed are one
budget.** Altitude is stored energy, speed is moving energy. Pitch never
adds or removes energy; it only moves it between the two accounts. Throttle
is the only input. Drag is the only output.

That gives the four-quadrant matrix that governs every phase:

|                 | Too fast                          | Slow / on speed                        |
|-----------------|-----------------------------------|----------------------------------------|
| **Too high**    | Total energy excess: cut power, add drag, bleed | Still excess (altitude is energy): idle, let the sink develop |
| **Too low**     | Split problem: pitch up, trade speed back into altitude | The dangerous corner: power is the ONLY fix |

Rules derived from the matrix, each now enforced in code:

- **Altitude is religion.** Altitude targets outrank speed and everything
  else. Above the line, the only acceptable state change is downward, and
  the fix for slow-while-high is pitch down (a free conversion), never
  power. One exception by decree: the flare, where vertical speed replaces
  altitude as the sacred number, because at thirty feet the altitude IS the
  ground.
- **One physics veto:** on short final, slow gets power regardless of the
  slope. At seventy knots with full flaps the elevator physically gives up;
  near the ground, airspeed is the flare.
- **The engine is not a switch.** Closed-loop throttle slews; only
  explicit ribbon orders (takeoff power, flare idle) are instant. The one
  counter-exception is stall recovery, where the engine IS a switch.
- **Configure early.** All drag comes out at the top of the descent (gear
  immediately, flaps staged by speed within seconds), where there is
  altitude to absorb the disturbance. Every flap event that ever went wrong
  went wrong mid-descent. With the drag out, the engine stays spooled
  against it and the throttle has authority in both directions.
- **Flaps are speed-gated, always.** Above flap-safe speed nothing new
  deploys, in any phase including the flare; past a structural margin
  anything out gets pulled back in. The altitude logic decides whether
  flaps are wanted; the speed gate decides whether they are safe.
- **Bleeding never climbs.** Pitching up to shed speed while already
  climbing just re-banks the energy for a later, worse withdrawal (one
  flight zoomed to a 20-knot tailslide doing this).

---

## 4. The Control Stack

**Lateral:** an L1 follower chases the ribbon polyline with a
speed-scaled lookahead. Cross-track capture and turn anticipation fall out
of the geometry. On the ground, steering switches to meter-scale
centerline tracking (on a 45-meter runway, miles are the wrong unit), with
an integrator to kill steady offsets that proportional control can only
balance against, never remove.

**Vertical: the cascade, and why it must exist.** The stick is not a
vertical speed lever. Physically, stick position maps to pitch *rate*; two
integrations and one to two seconds of aerodynamic lag separate the stick
from measured vertical speed. Proportional control across a double
integrator oscillates at every gain: soft gain gives a slow porpoise,
strong gain gives full-scale railing. Both were flown, repeatedly, before
the structure changed. The cascade:

- **Inner loop:** the stick holds pitch *attitude* (one integration,
  stiff, damped by pitch-rate feedback).
- **Outer loop:** vertical-speed error nudges the attitude target by
  fractions of a degree and a slow trim integrator finds the attitude that
  holds the slope. Filtered input, a deadband where close enough means
  touch nothing, and a creep-limited target: hold the attitude, correct in
  micro-adjustments.

**Damping, never filtering.** The hardest-won lesson in the project:
every smoothing layer placed inside a fast loop (a rate limiter, a heavy
low-pass) is phase delay, and enough delay turns the damping term into a
driver. The oscillation we chased for two days was manufactured by our own
smoothness patches. Smoothness is a *consequence* of well-timed damping,
never a decoration applied to the output.

**Continuity: each phase continues the last.** Emitted altitude and speed
commands are rate-limited at the engine, so a keyframe advance can never
step the plane's orders. Targets may jump; commands may not. (Near-ground
phases are exempt: there, commands must be instant truth.) The same
principle inside the controller: mode changes reset stale integrators, and
the attitude cascade initializes from the current attitude so even entering
a mode is seamless.

---

## 5. The Landing

Above the gate the plane negotiates; below it, it executes.

- **The stabilized gate** (modeled on autoland practice): commitment
  requires low, close, aligned, sink under control, speed sane. Unstable at
  the gate means no commitment and the approach laws keep flying.
  **[OPEN]** the honest response to an unstable gate is a go-around, which
  does not exist yet: today the plane keeps trying, which is how you get a
  firm uncommitted arrival.
- **The commitment latch** is one-way. Throttle retards and stays there;
  the sink target follows the flare curve as a ratchet (only ever
  shallower, so an altimeter blip cannot re-steepen it); nothing below the
  gate ever commands up; bank is capped against wing strikes.
- **Committed never means careless:** dangerous sink near the ground
  bypasses the ratchet for a full-authority arrest, and a balloon (climbing
  while committed) opens the nose-down cap, because stopping a balloon IS
  respecting vertical speed.
- **Touchdown is debounced** (one noisy altimeter frame once latched the
  rollout twenty-two feet in the air) and **bounces are handled**: if the
  rollout finds itself airborne again it flies a flare arrest, not the
  nosewheel-down push that once slammed a bouncing plane three times.
- **Derotation:** at wheels-down, gentle forward stick puts the nosewheel
  down and keeps it there; nose-up is hard-capped on the ground
  (tail-strike protection).
- **Rollout:** rudder holds the centerline (heading-hold alone let the
  plane weathervane off the pavement), brakes ramp progressively with
  speed.

---

## 6. The Protections Ledger

Every guard cites the flight that taught it. This table is the project's
crash history, inverted into armor.

| Protection | Taught by |
|---|---|
| Stall floor (low and slow: power, instantly) | Mushed 113→68 kts at idle above the slope |
| Alt-gate on the stall floor (high and slow: pitch, not power) | Throttle surge at 160 ft while trying to land |
| Short-final exception (slow below 600 ft: power, slope or not) | 70-kt final, elevator gave up, dove and bounced |
| Flap speed gate, all phases | Flaps at speed ballooned the plane away from the runway; recovery dive hit the ground |
| Configure-early descent | Every mid-descent flap event: balloons, lockouts, low config churn |
| Bleed never climbs | 30 kts traded for 300 ft on a level leg; separately, a zoom to a 20-kt tailslide |
| No diving for speed at/below target altitude | Transition dove for speed it should have bought with power, then zoomed |
| PID reset on coupling flips | Stale integrals fired full-stick surprises minutes after being frozen |
| Command continuity ramps | Descent trigger fired late and stepped the altitude target 300 ft in one tick |
| Along-route trigger gates | A loop cascaded through five phases in one tick on the outbound leg |
| End-flight replay guards | Stale broadcast reset the plane on every FLY ("click fly, it just reloads") |
| Touchdown debounce + bounce guard | Rollout latched 22 ft up on one bad frame; derotation mid-bounce slammed the plane |
| Emergency arrest inside commitment | Committed is not careless; deep sink near the ground gets full authority |
| True runway axis from endpoints | Stored heading off by up to 10°: grass takeoffs and landings beside the runway |
| Meter-scale ground steering + integrator | Whisper-gain steering "worked" while the plane mowed the grass |
| Damping-not-filtering | Two days of oscillation manufactured by our own smoothing layers |

---

## 7. Open Questions

1. **Go-around.** The stabilized gate can refuse, but refusal currently
   means "keep trying," not "climb, rejoin, retry." Design exists in
   outline; needs a ruling on when it may trigger autonomously.
2. **Cruise speed by trip length.** Plan-time speed selection, symmetric
   with the altitude picker. Blocked on one max-cruise calibration flight.
3. **X-Plane stability augmentation.** The SF50 model may run Garmin
   envelope protections that fight external yoke input at low speed. If
   residual wobble survives the cascade, this is the prime suspect; it
   would also not exist on the real airframe.
4. **Wind.** The follower commands heading, not ground track; calm-air
   sim hides the bias. The real airframe will not.
5. **Time-warp iteration.** Auto ×4 sim speed through trusted phases,
   ×1 latched from the deceleration point down. Designed, not built.
