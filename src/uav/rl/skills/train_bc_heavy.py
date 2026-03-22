"""Pure behavioral cloning with heavy synthetic augmentation.

No simulator needed. Trains in seconds. Generates 50K+ synthetic samples
covering every flight scenario: stable flight, heading corrections,
altitude corrections, bank recovery, combined errors, edge cases.

The key insight: for level flight, the correct control response is a
known function of the aircraft state. We don't need RL to discover it —
we can just teach it directly.

Observation (6):
    [0] pitch / 20
    [1] roll / 30
    [2] heading_error / 30
    [3] altitude_error / 200
    [4] speed_error / 20  (zeroed for level flight)
    [5] pitch / 10

Action (2):
    [0] pitch_cmd  [-1, 1]  → elevator (scaled by 0.5 in fly_model)
    [1] roll_cmd   [-1, 1]  → aileron  (scaled by 0.5 in fly_model)

Usage:
    python -m uav.rl.skills.train_bc_heavy --epochs 200
    python -m uav.rl.skills.fly_model models/level_flight_bc.npz
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger(__name__)


def generate_synthetic_dataset(n_samples: int = 50000, seed: int = 42) -> tuple:
    """Generate a massive synthetic dataset of (observation, correct_action) pairs.

    Covers:
    1. Stable flight (near-zero errors → near-zero commands)
    2. Heading corrections (heading error → roll to correct)
    3. Altitude corrections (altitude error → pitch to correct)
    4. Bank recovery (excessive roll → roll command to level wings)
    5. Pitch recovery (excessive pitch → pitch command to level)
    6. Combined errors (realistic multi-axis deviations)
    7. Edge cases (large errors, near-limits)
    """
    rng = np.random.default_rng(seed)
    obs_list = []
    act_list = []

    def _add(pitch_norm, roll_norm, hdg_err_norm, alt_err_norm, spd_err_norm, pitch2_norm,
             pitch_cmd, roll_cmd):
        obs_list.append([pitch_norm, roll_norm, hdg_err_norm, alt_err_norm, spd_err_norm, pitch2_norm])
        act_list.append([
            np.clip(pitch_cmd, -1.0, 1.0),
            np.clip(roll_cmd, -1.0, 1.0),
        ])

    # ── Control law parameters ──
    # These define the "expert" controller the BC will learn to mimic.
    # Tuned for smooth, proportional responses.

    # Heading correction: roll into heading error
    # hdg_err_norm = (current - target) / 30. If positive → right of target → roll LEFT (negative)
    K_HDG_ROLL = -1.5   # roll gain for heading error

    # Altitude correction: pitch to correct altitude error
    # alt_err_norm = (current - target) / 200. If positive → too high → pitch DOWN (positive pitch_cmd pushes nose down? No...)
    # Actually: pitch_cmd > 0 = nose up, pitch_cmd < 0 = nose down
    # If too high (alt_err > 0), pitch down → pitch_cmd negative
    K_ALT_PITCH = -0.8  # pitch gain for altitude error

    # Bank recovery: if rolled, command opposite roll to level
    # roll_norm = roll_deg / 30. If positive (right bank), need negative roll_cmd to level
    K_BANK_RECOVERY = -0.5  # roll gain for current bank angle

    # Pitch recovery: if pitched away from level, correct
    # pitch_norm = pitch_deg / 20. Level flight ~ slight nose up (1-3°)
    # Target pitch ~ 0.1 (2° nose up). If pitch > target → push nose down
    TARGET_PITCH_NORM = 0.1  # ~2° nose up for level flight
    K_PITCH_RECOVERY = -0.6  # pitch gain for pitch error

    samples_per_category = n_samples // 7

    # ── 1. Stable flight (near-zero everything) ──
    log.info("Generating stable flight samples...")
    for _ in range(samples_per_category):
        pitch = rng.normal(0.1, 0.05)   # ~2° nose up, small variation
        roll = rng.normal(0.0, 0.03)    # wings nearly level
        hdg_err = rng.normal(0.0, 0.02) # on heading
        alt_err = rng.normal(0.0, 0.02) # on altitude
        pitch2 = pitch * 2.0            # obs[5] = pitch/10 = (pitch*20)/10

        # Near-zero corrections
        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)
        roll_cmd = K_HDG_ROLL * hdg_err + K_BANK_RECOVERY * roll

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 2. Heading corrections (wide range) ──
    log.info("Generating heading correction samples...")
    for _ in range(samples_per_category):
        hdg_err = rng.uniform(-1.0, 1.0)  # ±30° heading error
        # Current state: mostly level with some roll from the error
        roll = rng.normal(hdg_err * 0.3, 0.1)  # drifting roll
        pitch = rng.normal(0.1, 0.08)
        alt_err = rng.normal(0.0, 0.1)
        pitch2 = pitch * 2.0

        roll_cmd = K_HDG_ROLL * hdg_err + K_BANK_RECOVERY * roll
        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 3. Altitude corrections (wide range) ──
    log.info("Generating altitude correction samples...")
    for _ in range(samples_per_category):
        alt_err = rng.uniform(-1.0, 1.0)  # ±200ft altitude error
        pitch = rng.normal(0.1 - alt_err * 0.1, 0.08)  # pitch affected by alt error
        roll = rng.normal(0.0, 0.05)
        hdg_err = rng.normal(0.0, 0.05)
        pitch2 = pitch * 2.0

        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)
        roll_cmd = K_HDG_ROLL * hdg_err + K_BANK_RECOVERY * roll

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 4. Bank recovery (excessive roll) ──
    log.info("Generating bank recovery samples...")
    for _ in range(samples_per_category):
        roll = rng.uniform(-1.5, 1.5)  # ±45° bank
        hdg_err = rng.normal(roll * 0.3, 0.1)  # heading drifting due to bank
        pitch = rng.normal(0.1, 0.1)
        alt_err = rng.normal(0.0, 0.15)
        pitch2 = pitch * 2.0

        # Strong roll correction + heading correction
        roll_cmd = K_BANK_RECOVERY * roll + K_HDG_ROLL * hdg_err * 0.5  # prioritize leveling wings
        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 5. Pitch recovery (excessive pitch) ──
    log.info("Generating pitch recovery samples...")
    for _ in range(samples_per_category):
        pitch = rng.uniform(-0.5, 0.75)  # -10° to +15° pitch
        alt_err = rng.normal(-pitch * 0.5, 0.1)  # altitude affected by pitch
        roll = rng.normal(0.0, 0.05)
        hdg_err = rng.normal(0.0, 0.05)
        pitch2 = pitch * 2.0

        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)
        roll_cmd = K_HDG_ROLL * hdg_err + K_BANK_RECOVERY * roll

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 6. Combined errors (realistic multi-axis) ──
    log.info("Generating combined error samples...")
    for _ in range(samples_per_category):
        hdg_err = rng.uniform(-0.8, 0.8)
        alt_err = rng.uniform(-0.8, 0.8)
        roll = rng.uniform(-1.0, 1.0)
        pitch = rng.uniform(-0.3, 0.5)
        pitch2 = pitch * 2.0

        roll_cmd = K_HDG_ROLL * hdg_err + K_BANK_RECOVERY * roll
        pitch_cmd = K_ALT_PITCH * alt_err + K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM)

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    # ── 7. Edge cases (large errors, near limits) ──
    log.info("Generating edge case samples...")
    for _ in range(samples_per_category):
        # Pick one or two channels to be extreme
        hdg_err = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]) + rng.normal(0, 0.1)
        alt_err = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]) + rng.normal(0, 0.1)
        roll = rng.uniform(-2.0, 2.0)  # ±60° bank — extreme
        pitch = rng.uniform(-0.75, 1.0)  # -15° to +20° pitch
        pitch2 = pitch * 2.0

        # Strong corrections — prioritize safety
        roll_cmd = K_BANK_RECOVERY * roll + K_HDG_ROLL * hdg_err * 0.3  # wings level first
        pitch_cmd = K_PITCH_RECOVERY * (pitch - TARGET_PITCH_NORM) + K_ALT_PITCH * alt_err * 0.3  # level pitch first

        _add(pitch, roll, hdg_err, alt_err, 0.0, pitch2, pitch_cmd, roll_cmd)

    obs = np.array(obs_list, dtype=np.float32)
    acts = np.array(act_list, dtype=np.float32)

    # Clip observations to match env range
    obs = np.clip(obs, -3.0, 3.0)

    log.info("Generated %d synthetic samples across 7 categories", len(obs))
    log.info("  Obs range: [%.2f, %.2f]", obs.min(), obs.max())
    log.info("  Act range: [%.2f, %.2f]", acts.min(), acts.max())
    log.info("  Act mean:  pitch=%.3f roll=%.3f", acts[:, 0].mean(), acts[:, 1].mean())
    log.info("  Act std:   pitch=%.3f roll=%.3f", acts[:, 0].std(), acts[:, 1].std())

    return obs, acts


def train_bc(obs: np.ndarray, acts: np.ndarray, epochs: int = 200,
             lr: float = 1e-3, batch_size: int = 256) -> nn.Sequential:
    """Train MLP policy via supervised learning."""
    obs_dim = obs.shape[1]
    act_dim = acts.shape[1]

    policy = nn.Sequential(
        nn.Linear(obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, act_dim),
        nn.Tanh(),
    )

    obs_t = torch.FloatTensor(obs)
    acts_t = torch.FloatTensor(acts)
    dataset = TensorDataset(obs_t, acts_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    log.info("Training BC for %d epochs on %d samples...", epochs, len(obs))
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_acts in loader:
            pred = policy(batch_obs)
            loss = loss_fn(pred, batch_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info("  Epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg_loss)

    # Verify: test on some known scenarios
    log.info("Verifying learned responses...")
    policy.eval()
    with torch.no_grad():
        tests = [
            ("Level flight (perfect)",     [0.1, 0.0, 0.0, 0.0, 0.0, 0.2]),
            ("Heading +15° right",         [0.1, 0.0, 0.5, 0.0, 0.0, 0.2]),
            ("Heading -15° left",          [0.1, 0.0, -0.5, 0.0, 0.0, 0.2]),
            ("100ft too high",             [0.1, 0.0, 0.0, 0.5, 0.0, 0.2]),
            ("100ft too low",              [0.1, 0.0, 0.0, -0.5, 0.0, 0.2]),
            ("Banked 15° right",           [0.1, 0.5, 0.0, 0.0, 0.0, 0.2]),
            ("Banked 15° left",            [0.1, -0.5, 0.0, 0.0, 0.0, 0.2]),
            ("Hdg +10° & 50ft high",       [0.1, 0.1, 0.33, 0.25, 0.0, 0.2]),
            ("Banked 30° right, hdg +20°", [0.1, 1.0, 0.67, 0.0, 0.0, 0.2]),
        ]
        for name, obs_vals in tests:
            inp = torch.FloatTensor([obs_vals])
            out = policy(inp).numpy()[0]
            log.info("  %-35s → pitch=%+.2f  roll=%+.2f", name, out[0], out[1])

    return policy


def export_to_npz(policy: nn.Sequential, output_path: str):
    """Export weights to .npz for NumpyMLPPolicy."""
    weights = {}
    layer_idx = 0
    for name, param in policy.named_parameters():
        p = param.detach().cpu().numpy()
        if "weight" in name:
            weights[f"w{layer_idx}"] = p.T  # PyTorch (out, in) → numpy (in, out)
        elif "bias" in name:
            weights[f"b{layer_idx}"] = p
            layer_idx += 1

    np.savez(output_path, **weights)
    log.info("Exported %d layers to %s", layer_idx, output_path)


def main():
    p = argparse.ArgumentParser(description="Train pure BC with heavy augmentation")
    p.add_argument("--samples", type=int, default=50000, help="Total synthetic samples")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-dir", default="models")
    args = p.parse_args()

    # Generate synthetic dataset
    obs, acts = generate_synthetic_dataset(n_samples=args.samples)

    # Also mix in real demo data if available
    demo_path = Path("demos/level_flight.npz")
    if demo_path.exists():
        data = np.load(demo_path)
        demo_obs = data["observations"]
        demo_acts = data["actions"][:, :2]  # pitch + roll only
        demo_obs[:, 4] = 0.0  # zero speed error
        obs = np.concatenate([obs, demo_obs], axis=0)
        acts = np.concatenate([acts, demo_acts], axis=0)
        log.info("Added %d real demo samples (total: %d)", len(demo_obs), len(obs))

    # Train
    policy = train_bc(obs, acts, epochs=args.epochs, lr=args.lr)

    # Export
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    export_to_npz(policy, str(save_dir / "level_flight_bc.npz"))

    # Also save as latest so fly_model picks it up
    export_to_npz(policy, str(save_dir / "level_flight_latest.npz"))

    log.info("Done! Test with: python -m uav.rl.skills.fly_model")


if __name__ == "__main__":
    main()
