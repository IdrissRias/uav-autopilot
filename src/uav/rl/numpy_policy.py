"""Lightweight MLP policy that runs inference with pure numpy.

No dependency on stable-baselines3, PyTorch, or ONNX — just numpy.
This is what ships to embedded hardware.

Weights are stored as a flat .npz file produced by ``export.py``.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path


class NumpyMLPPolicy:
    """Feed-forward MLP with tanh hidden activations.

    Parameters
    ----------
    weights_path : str | Path
        Path to an ``.npz`` file containing arrays named
        ``w0``, ``b0``, ``w1``, ``b1``, ... for each layer.
        The last layer has NO activation (raw logits → action).
    """

    def __init__(self, weights_path: str | Path) -> None:
        data = np.load(weights_path)
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        i = 0
        while f"w{i}" in data:
            w = data[f"w{i}"].astype(np.float32)
            b = data[f"b{i}"].astype(np.float32)
            self.layers.append((w, b))
            i += 1
        if not self.layers:
            raise ValueError(f"No weight arrays (w0, b0, ...) found in {weights_path}")

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Run a forward pass.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
            Normalised observation vector.

        Returns
        -------
        np.ndarray, shape (act_dim,)
            Raw action in [-1, 1] (tanh-squashed).
        """
        x = obs.astype(np.float32)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)  # hidden activation
        # Final layer: tanh squash to [-1, 1]
        return np.tanh(x)
