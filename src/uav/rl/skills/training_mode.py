"""Interactive training mode — learn to fly from human demonstrations.

The app guides you through flight maneuvers, records your inputs per skill,
trains instantly via behavioral cloning, then lets the AI try.

Usage:
    python -m uav.rl.skills.training_mode
    python -m uav.rl.skills.training_mode --skip-to cruise
    python -m uav.rl.skills.training_mode --chain-only
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
from uav.sim.types import Actuators, Telemetry
from uav.rl.skills.skill_registry import SkillDef, Phase, build_skill_registry, CONTROL_DREFS
from uav.rl.skills.phase_detector import PhaseDetector
from uav.rl.skills.demo_recorder import DemoRecorder
from uav.rl.skills.skill_trainer import train_skill
from uav.rl.skills.skill_pilot import SkillPilot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models/skills")
DEMOS_DIR = Path("demos/skills")


class TrainingSession:
    """Orchestrates the guided training session."""

    def __init__(
        self,
        adapter: XPlaneRLAdapter,
        loop_hz: float = 10.0,
    ):
        self.adapter = adapter
        self.loop_hz = loop_hz
        self.skills = build_skill_registry()
        self.phase_detector = PhaseDetector()
        self.trained_models: dict[str, Path] = {}
        self.targets: dict[str, float] = {}  # alt, hdg, spd — set during session

        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\n[TRAINING] Interrupted. Stopping gracefully...")
        self._running = False

    # ── Main session loop ────────────────────────────────────────────

    def run(self, skip_to: str | None = None, chain_only: bool = False) -> None:
        """Run the full interactive training session."""
        self._print_welcome()
        self._wait_for_telemetry()
        self._capture_initial_targets()

        if chain_only:
            self._run_chain_flight()
            return

        skills_to_run = self.skills
        if skip_to:
            names = [s.name for s in self.skills]
            if skip_to in names:
                idx = names.index(skip_to)
                skills_to_run = self.skills[idx:]
                print(f"[TRAINING] Skipping to: {skip_to}")
            else:
                print(f"[TRAINING] Unknown skill '{skip_to}'. Available: {names}")
                return

        for skill in skills_to_run:
            if not self._running:
                break
            self._run_skill_cycle(skill)

        if self._running and len(self.trained_models) >= 2:
            print("\n" + "=" * 60)
            print("  ALL SKILLS TRAINED!")
            print(f"  Models saved: {list(self.trained_models.keys())}")
            print("=" * 60)
            ans = input("\nRun full AI chain flight? [y/n] ").strip().lower()
            if ans == "y":
                self._run_chain_flight()

        print("[TRAINING] Session complete.")

    # ── Single skill cycle ───────────────────────────────────────────

    def _run_skill_cycle(self, skill: SkillDef) -> None:
        """Prompt → Record → Train → AI Demo → Evaluate for one skill."""
        while self._running:
            # 1. Prompt
            prompt = skill.prompt.format(**self.targets)
            print(f"\n{'─' * 60}")
            print(f"  SKILL: {skill.name.upper()}")
            print(f"  {prompt}")
            print(f"{'─' * 60}")
            input("  Press ENTER when ready to start, then fly! ")

            # 2. Record
            obs, acts = self._record_skill(skill)
            if obs is None or len(obs) < 10:
                print(f"[TRAINING] Too few samples ({len(obs) if obs is not None else 0}). Try again.")
                continue

            print(f"\n[TRAINING] Recorded {len(obs)} samples in {len(obs)/self.loop_hz:.0f}s")
            self._print_demo_stats(obs, acts, skill)

            # Save demo
            demo_path = DEMOS_DIR / f"{skill.name}.npz"
            demo_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(demo_path), observations=obs, actions=acts,
                     skill_name=skill.name, **{f"target_{k}": v for k, v in self.targets.items()})

            # 3. Train + Report
            print(f"\n[TRAINING] Training {skill.name}...")
            weights_path, report = train_skill(obs, acts, skill, output_dir=str(MODELS_DIR))
            self.trained_models[skill.name] = weights_path

            # 4. AI Demo
            ans = input("\n  Let the AI try? [y/n/r=re-record] ").strip().lower()
            if ans == "r":
                continue  # re-record
            if ans == "y":
                self._run_ai_demo(skill)

            # 5. Evaluate
            ans = input("\n  Satisfied with this skill? [y/n=re-record] ").strip().lower()
            if ans != "n":
                # Update targets based on current state (for next skill)
                self._update_targets_from_telemetry()
                break  # next skill

    # ── Recording ────────────────────────────────────────────────────

    def _record_skill(self, skill: SkillDef) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Record human demo for one skill."""
        recorder = DemoRecorder(self.adapter, skill, loop_hz=self.loop_hz)
        recorder.subscribe_control_drefs()

        print(f"[RECORDING] Started — fly the maneuver. Press ENTER to stop.")
        recorder.start()

        tick = 0
        min_samples = int(skill.min_duration_s * self.loop_hz)
        completed = False

        # Non-blocking input check
        import select
        import sys as _sys

        while self._running:
            loop_start = time.time()

            t = self.adapter.read_telemetry()
            extras = self.phase_detector.get_extras()
            self.phase_detector.update(t)

            # Also read vertical speed from dref 205 if available
            extras["throttle_pos"] = 0.0  # will be set by recorder.tick

            recorder.tick(t, self.targets, extras)
            tick += 1

            # Status update every 2 seconds
            if tick % (int(self.loop_hz) * 2) == 0:
                agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
                cr = extras.get("climb_rate_fpm", 0.0)
                print(f"  [{recorder.n_samples} samples] alt={t.altitude_ft:.0f} "
                      f"hdg={t.heading_deg:.0f} spd={t.airspeed_kts:.0f} "
                      f"agl={agl_ft:.0f} cr={cr:.0f}fpm "
                      f"phase={self.phase_detector.phase.value}")

            # Check completion
            if recorder.n_samples >= min_samples and skill.completion_check:
                if skill.completion_check(t, self.targets, extras):
                    if not completed:
                        completed = True
                        print(f"\n[RECORDING] Maneuver complete! Press ENTER to stop recording.")

            # Check for Enter key (non-blocking)
            if _sys.stdin in select.select([_sys.stdin], [], [], 0)[0]:
                _sys.stdin.readline()
                break

            # Rate limit
            dt = time.time() - loop_start
            sleep_time = (1.0 / self.loop_hz) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        return recorder.stop()

    # ── AI Demo ──────────────────────────────────────────────────────

    def _run_ai_demo(self, skill: SkillDef, duration: float = 30.0) -> None:
        """Let the AI fly one skill for a set duration."""
        weights_path = self.trained_models.get(skill.name)
        if not weights_path or not weights_path.exists():
            print(f"[AI] No trained model for {skill.name}")
            return

        pilot = SkillPilot(self.adapter, skill, str(weights_path))

        # Read current actuators for smooth blend
        t = self.adapter.read_telemetry()
        current_act = Actuators(
            throttle=skill.throttle_default,
            pitch=0.0, roll=0.0, yaw=0.0,
            gear_down=skill.gear_default,
            flap_ratio=skill.flap_default,
        )
        pilot.start_takeover(current_act)

        print(f"\n[AI] Taking over — flying {skill.name} for {duration:.0f}s...")
        print(f"[AI] Press ENTER to give control back to you.")

        import select
        import sys as _sys

        start = time.time()
        tick = 0

        while self._running and (time.time() - start) < duration:
            loop_start = time.time()

            t = self.adapter.read_telemetry()
            if not t.is_valid():
                time.sleep(0.05)
                continue

            extras = self.phase_detector.get_extras()
            self.phase_detector.update(t)

            # AI inference + blend
            actuators = pilot.tick(t, self.targets, extras)
            self.adapter.write_actuators(actuators)

            tick += 1
            if tick % (int(self.loop_hz) * 2) == 0:
                elapsed = time.time() - start
                agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
                blend = "blending" if pilot.is_blending else "AI"
                print(f"  [{elapsed:.0f}s] [{blend}] alt={t.altitude_ft:.0f} "
                      f"hdg={t.heading_deg:.0f} spd={t.airspeed_kts:.0f} "
                      f"agl={agl_ft:.0f} "
                      f"p={actuators.pitch:+.2f} r={actuators.roll:+.2f} "
                      f"thr={actuators.throttle:.2f}")

            # Check for Enter key
            if _sys.stdin in select.select([_sys.stdin], [], [], 0)[0]:
                _sys.stdin.readline()
                print("[AI] Giving control back to you.")
                break

            # Rate limit
            dt = time.time() - loop_start
            sleep_time = (1.0 / self.loop_hz) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[AI] Demo ended ({tick} steps).")

    # ── Chain Flight ─────────────────────────────────────────────────

    def _run_chain_flight(self) -> None:
        """AI flies the entire flight using phase detector to switch skills."""
        if not self.trained_models:
            # Try to load existing models
            for skill in self.skills:
                path = MODELS_DIR / f"{skill.name}.npz"
                if path.exists():
                    self.trained_models[skill.name] = path

        if not self.trained_models:
            print("[CHAIN] No trained models found. Train skills first.")
            return

        print(f"\n{'=' * 60}")
        print(f"  FULL AI CHAIN FLIGHT")
        print(f"  Available skills: {list(self.trained_models.keys())}")
        print(f"  Phase detector will switch between skills automatically.")
        print(f"  Press ENTER at any time to stop.")
        print(f"{'=' * 60}")
        input("  Press ENTER to start... ")

        # Build pilots for all trained skills
        pilots: dict[str, SkillPilot] = {}
        for skill in self.skills:
            if skill.name in self.trained_models:
                pilots[skill.name] = SkillPilot(
                    self.adapter, skill, str(self.trained_models[skill.name])
                )

        # Map phases to skill names
        phase_to_skill = {skill.phase: skill.name for skill in self.skills
                          if skill.name in pilots}

        import select
        import sys as _sys

        current_skill_name: str | None = None
        tick = 0

        while self._running:
            loop_start = time.time()

            t = self.adapter.read_telemetry()
            if not t.is_valid():
                time.sleep(0.05)
                continue

            # Detect phase
            extras = self.phase_detector.get_extras()
            phase = self.phase_detector.update(t)

            # Select skill for current phase
            skill_name = phase_to_skill.get(phase)

            if skill_name and skill_name != current_skill_name and skill_name in pilots:
                # Switch skill
                print(f"[CHAIN] Phase: {phase.value} → activating {skill_name}")
                current_act = Actuators(
                    throttle=0.5, pitch=0.0, roll=0.0, yaw=0.0,
                    gear_down=False, flap_ratio=0.0,
                )
                pilots[skill_name].start_takeover(current_act)
                current_skill_name = skill_name

            # Run current skill
            if current_skill_name and current_skill_name in pilots:
                actuators = pilots[current_skill_name].tick(t, self.targets, extras)
                self.adapter.write_actuators(actuators)

            tick += 1
            if tick % (int(self.loop_hz) * 3) == 0:
                agl_ft = t.agl_m * 3.28084 if not math.isnan(t.agl_m) else 0.0
                print(f"  [{phase.value:>8}] [{current_skill_name or 'none':>8}] "
                      f"alt={t.altitude_ft:.0f} hdg={t.heading_deg:.0f} "
                      f"spd={t.airspeed_kts:.0f} agl={agl_ft:.0f}")

            # Check for Enter key
            if _sys.stdin in select.select([_sys.stdin], [], [], 0)[0]:
                _sys.stdin.readline()
                print("[CHAIN] Stopped. You have control.")
                break

            dt = time.time() - loop_start
            sleep_time = (1.0 / self.loop_hz) - dt
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"[CHAIN] Flight ended ({tick} steps).")

    # ── Helpers ──────────────────────────────────────────────────────

    def _wait_for_telemetry(self) -> None:
        """Wait until X-Plane is sending valid telemetry."""
        print("[TRAINING] Waiting for X-Plane telemetry...", end="", flush=True)
        for _ in range(60):
            t = self.adapter.read_telemetry()
            if t.is_valid():
                print(" connected!")
                return
            time.sleep(0.5)
            print(".", end="", flush=True)
        print("\n[TRAINING] WARNING: No valid telemetry after 30s. Continuing anyway.")

    def _capture_initial_targets(self) -> None:
        """Capture current aircraft state as session targets."""
        t = self.adapter.read_telemetry()
        self.targets = {
            "alt": t.altitude_ft,
            "hdg": t.heading_deg,
            "spd": t.airspeed_kts,
        }
        print(f"[TRAINING] Session targets: alt={t.altitude_ft:.0f}ft "
              f"hdg={t.heading_deg:.0f}° spd={t.airspeed_kts:.0f}kts")

    def _update_targets_from_telemetry(self) -> None:
        """Update targets to current aircraft state (for next skill)."""
        t = self.adapter.read_telemetry()
        if t.is_valid():
            self.targets["alt"] = t.altitude_ft
            self.targets["hdg"] = t.heading_deg
            self.targets["spd"] = t.airspeed_kts

    def _print_demo_stats(self, obs: np.ndarray, acts: np.ndarray, skill: SkillDef) -> None:
        """Print summary statistics for recorded demo."""
        print(f"  Observations: {obs.shape}")
        print(f"  Actions:      {acts.shape}")
        for i, name in enumerate(skill.act_names):
            col = acts[:, i]
            print(f"    {name:>10}: mean={col.mean():+.3f}  std={col.std():.3f}  "
                  f"range=[{col.min():+.2f}, {col.max():+.2f}]")

    def _print_welcome(self) -> None:
        print()
        print("=" * 60)
        print("  INTERACTIVE FLIGHT TRAINING MODE")
        print("  ─────────────────────────────────")
        print("  You fly. The AI watches and learns.")
        print("  After each maneuver, the AI trains instantly")
        print("  and tries to replicate what you did.")
        print()
        print("  Skills: takeoff → climb → turn → cruise → descend → land")
        print("=" * 60)
        print()


def main():
    p = argparse.ArgumentParser(description="Interactive flight training mode")
    p.add_argument("--xplane-ip", default="127.0.0.1")
    p.add_argument("--xplane-port", type=int, default=49000)
    p.add_argument("--loop-hz", type=float, default=10.0)
    p.add_argument("--skip-to", type=str, default=None,
                   help="Skip to a specific skill (e.g., cruise, land)")
    p.add_argument("--chain-only", action="store_true",
                   help="Skip training, go straight to chain flight with existing models")
    args = p.parse_args()

    adapter = XPlaneRLAdapter(
        xplane_ip=args.xplane_ip,
        xplane_port=args.xplane_port,
        local_port=0,
        freq_hz=int(args.loop_hz),
    )
    time.sleep(0.5)

    session = TrainingSession(adapter=adapter, loop_hz=args.loop_hz)
    session.run(skip_to=args.skip_to, chain_only=args.chain_only)


if __name__ == "__main__":
    main()
