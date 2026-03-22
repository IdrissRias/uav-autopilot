"""Export SB3 model weights to a lightweight .npz file for numpy inference.

Usage:
    python -m uav.rl.export path/to/model.zip path/to/output.npz
"""
from __future__ import annotations

import numpy as np


def export_sb3_to_numpy(
    model_or_path,
    output_path: str,
) -> None:
    """Extract MLP weights from an SB3 model and save as .npz.

    Parameters
    ----------
    model_or_path : str | PPO | SAC
        Either a file path to a saved SB3 model (.zip), or an already-loaded
        SB3 model object.
    output_path : str
        Where to write the .npz file.
    """
    if isinstance(model_or_path, str):
        # Try SAC first, fall back to PPO
        try:
            from stable_baselines3 import SAC
            model = SAC.load(model_or_path)
        except Exception:
            from stable_baselines3 import PPO
            model = PPO.load(model_or_path)
    else:
        model = model_or_path

    policy = model.policy
    weights = {}
    layer_idx = 0

    # Detect architecture: SAC vs PPO
    is_sac = hasattr(policy, 'actor') and hasattr(policy.actor, 'latent_pi')

    if is_sac:
        # SAC: actor.latent_pi (hidden layers) + actor.mu (mean action layer)
        for name, param in policy.actor.latent_pi.named_parameters():
            p = param.detach().cpu().numpy()
            if "weight" in name:
                weights[f"w{layer_idx}"] = p.T
            elif "bias" in name:
                weights[f"b{layer_idx}"] = p
                layer_idx += 1

        for name, param in policy.actor.mu.named_parameters():
            p = param.detach().cpu().numpy()
            if "weight" in name:
                weights[f"w{layer_idx}"] = p.T
            elif "bias" in name:
                weights[f"b{layer_idx}"] = p
    else:
        # PPO: mlp_extractor.policy_net (hidden layers) + action_net (output layer)
        for name, param in policy.mlp_extractor.policy_net.named_parameters():
            p = param.detach().cpu().numpy()
            if "weight" in name:
                weights[f"w{layer_idx}"] = p.T
            elif "bias" in name:
                weights[f"b{layer_idx}"] = p
                layer_idx += 1

        for name, param in policy.action_net.named_parameters():
            p = param.detach().cpu().numpy()
            if "weight" in name:
                weights[f"w{layer_idx}"] = p.T
            elif "bias" in name:
                weights[f"b{layer_idx}"] = p

    np.savez(output_path, **weights)
    print(f"Exported {layer_idx + 1} layers to {output_path}")
    for k, v in weights.items():
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python -m uav.rl.export <model.zip> <output.npz>")
        sys.exit(1)
    export_sb3_to_numpy(sys.argv[1], sys.argv[2])
