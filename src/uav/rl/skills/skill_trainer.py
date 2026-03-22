"""Fast behavioral cloning training per skill.

Trains a small MLP from demo data in ~3 seconds on CPU.
Uses noise augmentation to expand small datasets.
Exports to .npz compatible with NumpyMLPPolicy.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from uav.rl.skills.skill_registry import SkillDef

log = logging.getLogger(__name__)


def augment_with_noise(
    obs: np.ndarray,
    acts: np.ndarray,
    copies: int = 5,
    obs_sigma: float = 0.02,
    act_sigma: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand dataset with gaussian noise copies."""
    rng = np.random.default_rng(seed)
    aug_obs = [obs]
    aug_acts = [acts]

    for _ in range(copies):
        noisy_obs = obs + rng.normal(0, obs_sigma, obs.shape).astype(np.float32)
        noisy_acts = acts + rng.normal(0, act_sigma, acts.shape).astype(np.float32)
        noisy_obs = np.clip(noisy_obs, -3.0, 3.0)
        noisy_acts = np.clip(noisy_acts, -1.0, 1.0)
        aug_obs.append(noisy_obs)
        aug_acts.append(noisy_acts)

    return np.concatenate(aug_obs, axis=0), np.concatenate(aug_acts, axis=0)


def train_skill(
    obs: np.ndarray,
    acts: np.ndarray,
    skill: SkillDef,
    output_dir: str | Path = "models/skills",
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    augment: bool = True,
) -> Path:
    """Train BC for a single skill. Returns path to saved .npz weights.

    Parameters
    ----------
    obs : (N, obs_dim) observation array
    acts : (N, act_dim) action array
    skill : SkillDef with obs_dim and act_dim
    output_dir : directory for saving weights
    epochs : training epochs
    lr : learning rate
    batch_size : mini-batch size
    augment : whether to apply noise augmentation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("[%s] Training on %d raw samples (obs=%dD, act=%dD)",
             skill.name, len(obs), skill.obs_dim, skill.act_dim)

    # Augment
    if augment and len(obs) > 0:
        obs, acts = augment_with_noise(obs, acts, copies=5)
        log.info("[%s] After augmentation: %d samples", skill.name, len(obs))

    if len(obs) == 0:
        log.warning("[%s] No training data!", skill.name)
        return output_dir / f"{skill.name}.npz"

    # Build MLP: [obs_dim] → [128, ReLU] → [128, ReLU] → [act_dim, Tanh]
    policy = nn.Sequential(
        nn.Linear(skill.obs_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, skill.act_dim),
        nn.Tanh(),
    )

    dataset = TensorDataset(torch.FloatTensor(obs), torch.FloatTensor(acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for batch_obs, batch_acts in loader:
            pred = policy(batch_obs)
            loss = loss_fn(pred, batch_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info("[%s] Epoch %d/%d  loss=%.6f", skill.name, epoch + 1, epochs, avg_loss)

    # Export to .npz
    weights_path = output_dir / f"{skill.name}.npz"
    _export_to_npz(policy, str(weights_path))
    log.info("[%s] Saved to %s (loss=%.6f)", skill.name, weights_path, avg_loss)

    # Generate learning report
    report = generate_report(policy, skill, obs, acts)

    return weights_path, report


def _export_to_npz(policy: nn.Sequential, output_path: str) -> None:
    """Export PyTorch MLP weights to .npz for NumpyMLPPolicy."""
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


# ─── Scenario test definitions per skill ─────────────────────────────

# Each scenario: (description, obs_vector)
# Obs meanings vary by skill — see skill_registry.py for normalization

_SCENARIOS = {
    "takeoff": [
        ("On ground, full throttle, starting roll",  [0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]),
        ("Rotating — nose up 5°, 60kts",             [0.25, 0.0, 0.3, 0.05, 0.0, 0.8, 0.5]),
        ("Climbing — nose up 10°, 100ft AGL",        [0.5, 0.0, 0.5, 1.0, 0.0, 0.9, 1.0]),
        ("Drifting right 5° on takeoff",             [0.25, 0.1, 0.17, 0.5, 0.0, 0.9, 0.5]),
    ],
    "climb": [
        ("Steady climb, on heading, on target",      [0.25, 0.0, 0.0, -0.5, 0.5, 0.5, 0.8]),
        ("500ft below target, climbing",             [0.25, 0.0, 0.0, -1.0, 0.5, 0.5, 0.8]),
        ("Near target altitude, leveling off",       [0.1, 0.0, 0.0, -0.1, 0.5, 0.1, 0.7]),
        ("Heading drifted 10° right during climb",   [0.25, 0.1, 0.33, -0.5, 0.5, 0.5, 0.8]),
    ],
    "turn": [
        ("Need to turn 30° right",                   [0.05, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0]),
        ("Need to turn 30° left",                    [0.05, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0]),
        ("In a 15° right bank, 15° to go",           [0.05, 0.5, 0.5, 0.0, 0.5, 0.5, 2.0]),
        ("Almost on heading, rolling wings level",    [0.05, 0.1, 0.1, 0.0, 0.5, 0.1, 0.3]),
    ],
    "cruise": [
        ("Perfect level flight",                      [0.1, 0.0, 0.0, 0.0, 0.0, 0.2]),
        ("Heading 15° right of target",               [0.1, 0.0, 0.5, 0.0, 0.0, 0.2]),
        ("Heading 15° left of target",                [0.1, 0.0, -0.5, 0.0, 0.0, 0.2]),
        ("100ft too high",                            [0.1, 0.0, 0.0, 0.5, 0.0, 0.2]),
        ("100ft too low",                             [0.1, 0.0, 0.0, -0.5, 0.0, 0.2]),
        ("Banked 15° right, heading drifting",        [0.1, 0.5, 0.3, 0.0, 0.0, 0.2]),
    ],
    "descend": [
        ("Starting descent, reducing power",          [0.0, 0.0, 0.0, -0.5, 0.5, -0.3, 0.4]),
        ("In descent, 200ft above target",            [-.05, 0.0, 0.0, -0.4, 0.5, -0.4, 0.3]),
        ("Near target altitude, leveling off",        [0.0, 0.0, 0.0, -0.1, 0.5, -0.1, 0.4]),
        ("Heading drifted during descent",            [-.05, 0.1, 0.33, -0.3, 0.5, -0.3, 0.3]),
    ],
    "land": [
        ("On approach, 500ft AGL, lined up",          [0.0, 0.0, 0.0, 2.5, 0.4, -0.5, 0.0, 0.3]),
        ("On approach, 200ft AGL",                    [-.05, 0.0, 0.0, 1.0, 0.4, -0.4, -.05, 0.2]),
        ("Short final, 50ft AGL, flaring",            [0.1, 0.0, 0.0, 0.25, 0.35, -0.2, 0.1, 0.1]),
        ("Drifting left on approach",                 [0.0, -0.2, -0.2, 1.0, 0.4, -0.4, 0.0, 0.2]),
    ],
}


def generate_report(
    policy: nn.Sequential,
    skill: SkillDef,
    obs: np.ndarray,
    acts: np.ndarray,
) -> str:
    """Generate a human-readable report of what the model learned.

    Returns the report as a string (also prints it).
    """
    policy.eval()
    lines: list[str] = []

    def _line(s: str = "") -> None:
        lines.append(s)

    _line()
    _line(f"{'═' * 60}")
    _line(f"  LEARNING REPORT: {skill.name.upper()}")
    _line(f"{'═' * 60}")

    # ── Training data summary ──
    _line()
    _line("  TRAINING DATA:")
    _line(f"    Samples:  {len(obs)} (raw) → {len(obs)} (after augmentation)")
    _line(f"    Obs dim:  {skill.obs_dim}D")
    _line(f"    Act dim:  {skill.act_dim}D — {skill.act_names}")
    _line()
    _line("    Action statistics from YOUR demo:")
    for i, name in enumerate(skill.act_names):
        col = acts[:, i] if i < acts.shape[1] else np.zeros(len(acts))
        _line(f"      {name:>10}: mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"range=[{col.min():+.2f}, {col.max():+.2f}]")

    # ── Scenario tests ──
    scenarios = _SCENARIOS.get(skill.name, [])
    if scenarios:
        _line()
        _line("  WHAT I LEARNED (scenario tests):")
        _line()

        with torch.no_grad():
            for desc, obs_vals in scenarios:
                inp = torch.FloatTensor([obs_vals[:skill.obs_dim]])
                out = policy(inp).numpy()[0]
                acts_str = "  ".join(f"{n}={v:+.3f}" for n, v in zip(skill.act_names, out))
                _line(f"    {desc}")
                _line(f"      → {acts_str}")
                _line()

    # ── Sensitivity analysis ──
    _line("  SENSITIVITY (how much each input affects output):")
    _line()
    with torch.no_grad():
        baseline = torch.zeros(1, skill.obs_dim)
        base_out = policy(baseline).numpy()[0]

        obs_labels = _get_obs_labels(skill.name)
        for i in range(skill.obs_dim):
            perturbed = baseline.clone()
            perturbed[0, i] = 1.0  # +1 standard unit
            pert_out = policy(perturbed).numpy()[0]
            deltas = pert_out - base_out

            if any(abs(d) > 0.01 for d in deltas):
                label = obs_labels[i] if i < len(obs_labels) else f"obs[{i}]"
                effects = "  ".join(
                    f"{n}{'↑' if d > 0 else '↓'}{abs(d):.2f}" if abs(d) > 0.01 else ""
                    for n, d in zip(skill.act_names, deltas)
                ).strip()
                _line(f"    If {label} increases → {effects}")

    # ── Confidence score ──
    _line()
    _line("  CONFIDENCE:")
    # Compute how well the model fits the training data
    with torch.no_grad():
        pred = policy(torch.FloatTensor(obs[:min(len(obs), 500)])).numpy()
        actual = acts[:min(len(acts), 500)]
        mse = float(np.mean((pred - actual) ** 2))
        # R² score
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - actual.mean(axis=0)) ** 2)
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))
        _line(f"    Fit quality (R²): {r2:.3f}  {'(excellent)' if r2 > 0.9 else '(good)' if r2 > 0.7 else '(needs more data)'}")
        _line(f"    Mean squared error: {mse:.6f}")

    _line()
    _line(f"{'═' * 60}")

    report = "\n".join(lines)
    print(report)
    return report


def _get_obs_labels(skill_name: str) -> list[str]:
    """Human-readable labels for observation channels."""
    return {
        "takeoff": ["pitch", "roll", "airspeed", "AGL", "heading_err", "throttle", "climb_rate"],
        "climb": ["pitch", "roll", "heading_err", "alt_err", "airspeed", "climb_rate", "throttle"],
        "turn": ["pitch", "roll", "heading_err", "alt_err", "airspeed", "bank", "turn_rate"],
        "cruise": ["pitch", "roll", "heading_err", "alt_err", "speed_err", "pitch_rate"],
        "descend": ["pitch", "roll", "heading_err", "alt_err", "airspeed", "climb_rate", "throttle"],
        "land": ["pitch", "roll", "heading_err", "AGL", "airspeed", "climb_rate", "pitch_rate", "throttle"],
    }.get(skill_name, [f"obs[{i}]" for i in range(10)])
