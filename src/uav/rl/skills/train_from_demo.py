"""Train from human flight demonstrations (behavioral cloning + RL fine-tune).

Step 1: Behavioral cloning — supervised learning to mimic the human pilot
Step 2: RL fine-tuning — improve beyond human with PPO

Usage:
    python -m uav.rl.skills.train_from_demo demos/level_flight.npz --rl-timesteps 50000
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


def augment_heading_corrections(obs: np.ndarray, acts: np.ndarray, n_augmented: int = 2000) -> tuple:
    """Synthesize training samples that teach heading correction via roll.

    The real demo often has near-zero heading error and near-zero roll,
    so the model never learns "if heading drifts right, bank left."
    We generate synthetic samples with artificial heading errors
    paired with the correct roll response.
    """
    rng = np.random.default_rng(42)
    aug_obs = []
    aug_acts = []

    for _ in range(n_augmented):
        # Pick a random real sample as a base
        idx = rng.integers(0, len(obs))
        o = obs[idx].copy()

        # Inject a heading error: obs[2] = heading_error / 30
        hdg_err_norm = rng.uniform(-0.5, 0.5)  # ±15° heading error
        o[2] = hdg_err_norm

        # Correct roll response: bank into the error to correct it
        # heading_error = current - target. If positive, we're right of target, need to turn LEFT
        # Turn left = negative roll. So roll_cmd = -K * heading_error
        roll_correction = np.clip(-1.5 * hdg_err_norm, -1.0, 1.0)

        # Also inject mild altitude error samples
        alt_err_norm = rng.uniform(-0.3, 0.3)  # ±60ft
        o[3] = alt_err_norm
        # Pitch up if below target (negative error), pitch down if above
        pitch_correction = np.clip(-0.8 * alt_err_norm, -1.0, 1.0)

        # Action: [pitch, roll] — no throttle
        a = np.array([pitch_correction, roll_correction], dtype=np.float32)

        aug_obs.append(o)
        aug_acts.append(a)

    aug_obs = np.array(aug_obs, dtype=np.float32)
    aug_acts = np.array(aug_acts, dtype=np.float32)

    log.info("Generated %d augmented samples (heading+altitude corrections)", n_augmented)
    return aug_obs, aug_acts


def behavioral_cloning(demo_path: str, epochs: int = 100, lr: float = 1e-3, batch_size: int = 64):
    """Train a policy network to mimic human demonstrations via supervised learning."""
    data = np.load(demo_path)
    obs = data["observations"]  # (N, 6)
    acts = data["actions"][:, :2]  # (N, 2) — pitch + roll only, drop throttle
    target_alt = float(data["target_alt"])
    target_hdg = float(data["target_hdg"])
    target_spd = float(data["target_spd"])

    log.info("Loaded %d samples from %s", len(obs), demo_path)
    log.info("Targets: alt=%.0fft hdg=%.0f° spd=%.0fkts", target_alt, target_hdg, target_spd)

    # Zero out speed error — for level flight, speed doesn't matter.
    # The demo had speed pegged at clip max the whole time, which biases the model.
    obs[:, 4] = 0.0
    log.info("Zeroed speed_error channel (obs[4]) — level flight ignores speed")

    # Augment with synthetic heading/altitude correction samples
    aug_obs, aug_acts = augment_heading_corrections(obs, acts, n_augmented=2000)
    obs = np.concatenate([obs, aug_obs], axis=0)
    acts = np.concatenate([acts, aug_acts], axis=0)
    log.info("Total training samples after augmentation: %d", len(obs))

    # Build MLP matching SAC architecture: [128, 128]
    obs_dim = obs.shape[1]
    act_dim = acts.shape[1]
    hidden = 128

    policy = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),  # SAC uses ReLU, not Tanh
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, act_dim),
        nn.Tanh(),  # output in [-1, 1]
    )

    # Train dataset
    obs_t = torch.FloatTensor(obs)
    acts_t = torch.FloatTensor(acts)
    dataset = TensorDataset(obs_t, acts_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    log.info("Training behavioral cloning for %d epochs...", epochs)
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("  Epoch %d/%d  loss=%.6f", epoch + 1, epochs, avg_loss)

    return policy, target_alt, target_hdg, target_spd


def export_bc_to_npz(policy: nn.Sequential, output_path: str):
    """Export behavioral cloning weights to .npz for NumpyMLPPolicy."""
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


class _SaveAndExportCallback:
    """SB3 callback that saves checkpoint + numpy export every N steps."""

    def __init__(self, save_freq: int, save_dir: Path, export_fn):
        from stable_baselines3.common.callbacks import BaseCallback

        self._save_freq = save_freq
        self._save_dir = save_dir
        self._export_fn = export_fn
        self._save_dir.mkdir(parents=True, exist_ok=True)
        (self._save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        class _Inner(BaseCallback):
            def __init__(inner_self, verbose=0):
                super().__init__(verbose)
                inner_self.save_freq = save_freq
                inner_self.save_dir = save_dir
                inner_self.export_fn = export_fn
                inner_self.n_calls_at_last_save = 0

            def _on_step(inner_self) -> bool:
                if inner_self.num_timesteps - inner_self.n_calls_at_last_save >= inner_self.save_freq:
                    inner_self.n_calls_at_last_save = inner_self.num_timesteps
                    step = inner_self.num_timesteps
                    # Save SB3 checkpoint
                    ckpt_path = inner_self.save_dir / "checkpoints" / f"level_flight_{step}_steps"
                    inner_self.model.save(str(ckpt_path))
                    # Export to numpy for fly_model
                    npz_path = inner_self.save_dir / "level_flight_latest.npz"
                    inner_self.export_fn(inner_self.model, str(npz_path))
                    # Also save as best (overwritten each time — always latest)
                    inner_self.model.save(str(inner_self.save_dir / "level_flight_latest"))
                    log.info("[SAVE] Step %d — checkpoint + numpy export saved", step)
                return True

        self.callback = _Inner()


def _find_latest_checkpoint(save_dir: Path) -> str | None:
    """Find the most recent checkpoint to resume from."""
    ckpt_dir = save_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("level_flight_*_steps.zip"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        return None
    # Return path without .zip extension (SB3 convention)
    return str(checkpoints[-1]).removesuffix(".zip")


def rl_finetune(
    bc_policy: nn.Sequential | None,
    target_alt: float,
    target_hdg: float,
    target_spd: float,
    timesteps: int = 50000,
    save_dir: str = "models",
    resume: bool = False,
):
    """Fine-tune with SAC in X-Plane (switched from PPO for 2-5x sample efficiency).

    SAC is off-policy — it stores all experience in a replay buffer and reuses it,
    extracting far more learning per expensive simulator step (~9 FPS).

    If resume=True, loads the latest checkpoint instead of BC weights.
    Saves checkpoint + numpy export every 2048 steps.
    Handles SIGINT/SIGTERM gracefully — saves before exit.
    """
    import signal
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from uav.rl.xplane_rl_adapter import XPlaneRLAdapter
    from uav.rl.skills.level_flight import LevelFlightEnv
    from uav.rl.export import export_sb3_to_numpy

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    adapter = XPlaneRLAdapter(
        xplane_ip="127.0.0.1", xplane_port=49000, local_port=0, freq_hz=10,
    )

    raw_env = LevelFlightEnv(
        adapter=adapter,
        target_alt_ft=target_alt,
        target_hdg=target_hdg,
        target_speed_kts=target_spd,
        sim_speed=3.0,   # 3x sim speed — hardware limit
        loop_hz=10.0,
        episode_steps=600,  # shorter episodes — faster curriculum progression
    )
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])
    env = VecNormalize(
        vec_env,
        norm_obs=False,    # we already normalize obs manually
        norm_reward=True,  # auto-scale rewards for stable value function
        clip_reward=10.0,
        gamma=0.995,
    )

    # Try to resume from checkpoint
    latest_ckpt = _find_latest_checkpoint(save_path) if resume else None

    if latest_ckpt:
        log.info("RESUMING from checkpoint: %s", latest_ckpt)
        model = SAC.load(latest_ckpt, env=env)
        model.verbose = 1
    else:
        log.info("Creating SAC model (off-policy, 2-5x more sample efficient than PPO)...")
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4,
            buffer_size=100_000,        # replay buffer — reuse every experience
            learning_starts=200,        # collect 200 steps before first update
            batch_size=256,             # large batches for stable gradients
            tau=0.005,                  # soft target update
            gamma=0.995,               # long horizon
            train_freq=1,              # update every single step
            gradient_steps=4,          # 4 gradient steps per env step — squeeze more from each sample
            ent_coef=0.01,             # LOW entropy — we already have good BC policy, don't explore randomly
            policy_kwargs={
                "net_arch": [128, 128],
            },
            verbose=1,
        )

        # Load BC weights into SAC's actor network
        if bc_policy is not None:
            bc_state = bc_policy.state_dict()
            actor = model.policy.actor
            # SAC actor: latent_pi = Sequential(Linear(6,128), ReLU, Linear(128,128), ReLU)
            #            mu = Linear(128, 3)
            # BC:        0=Linear(6,128), 1=ReLU, 2=Linear(128,128), 3=ReLU, 4=Linear(128,3), 5=Tanh
            with torch.no_grad():
                actor.latent_pi[0].weight.copy_(bc_state["0.weight"])
                actor.latent_pi[0].bias.copy_(bc_state["0.bias"])
                actor.latent_pi[2].weight.copy_(bc_state["2.weight"])
                actor.latent_pi[2].bias.copy_(bc_state["2.bias"])
                actor.mu.weight.copy_(bc_state["4.weight"])
                actor.mu.bias.copy_(bc_state["4.bias"])
            log.info("BC weights loaded into SAC actor — starts flying from demo knowledge.")

    # Graceful shutdown — save on SIGINT/SIGTERM
    def _emergency_save(signum, frame):
        log.info("[EMERGENCY SAVE] Signal %d received — saving model...", signum)
        model.save(str(save_path / "level_flight_emergency"))
        export_sb3_to_numpy(model, str(save_path / "level_flight_emergency.npz"))
        log.info("[EMERGENCY SAVE] Done. Exiting.")
        raw_env.close()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    # Checkpoint callback — saves every 2048 steps
    saver = _SaveAndExportCallback(save_freq=2048, save_dir=save_path, export_fn=export_sb3_to_numpy)

    log.info("Training for %d steps (SAC, 4x sim speed, saves every 2048 steps)...", timesteps)
    model.learn(total_timesteps=timesteps, callback=saver.callback)

    # Final save
    model.save(str(save_path / "level_flight_finetuned"))
    export_sb3_to_numpy(model, str(save_path / "level_flight_finetuned.npz"))
    log.info("Saved final model to %s", save_path)

    raw_env.close()


def main():
    p = argparse.ArgumentParser(description="Train from human demo")
    p.add_argument("demo", help="Path to demo .npz file")
    p.add_argument("--bc-epochs", type=int, default=100)
    p.add_argument("--bc-lr", type=float, default=1e-3)
    p.add_argument("--rl-timesteps", type=int, default=50000, help="0 = BC only, no RL fine-tune")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint (skip BC, load last saved PPO)")
    p.add_argument("--save-dir", default="models")
    args = p.parse_args()

    if args.resume:
        # Resume RL from checkpoint — skip BC entirely
        log.info("Resuming from last checkpoint...")
        data = np.load(args.demo)
        target_alt = float(data["target_alt"])
        target_hdg = float(data["target_hdg"])
        target_spd = float(data["target_spd"])
        rl_finetune(None, target_alt, target_hdg, target_spd,
                     timesteps=args.rl_timesteps, save_dir=args.save_dir, resume=True)
        log.info("All done!")
        return

    # Step 1: Behavioral cloning
    policy, target_alt, target_hdg, target_spd = behavioral_cloning(
        args.demo, epochs=args.bc_epochs, lr=args.bc_lr,
    )

    # Save BC model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    export_bc_to_npz(policy, str(save_dir / "level_flight_bc.npz"))

    # Step 2: RL fine-tuning (optional)
    if args.rl_timesteps > 0:
        log.info("Starting RL fine-tuning...")
        rl_finetune(policy, target_alt, target_hdg, target_spd,
                     timesteps=args.rl_timesteps, save_dir=args.save_dir)
    else:
        log.info("Skipping RL fine-tune (--rl-timesteps=0)")

    log.info("All done!")


if __name__ == "__main__":
    main()
