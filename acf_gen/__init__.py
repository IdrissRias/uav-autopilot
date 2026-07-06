"""acf_gen — generate flyable X-Plane aircraft from a parametric description.

X-Plane is the runtime: we emit a correct .acf (+ matching .obj) and X-Plane's
blade-element solver flies it. We build no physics, sim, or renderer of our own.

Stage 1 (this module): a lossless .acf round-trip reader/writer (``ACF``).
"""

from .acf import ACF

__all__ = ["ACF"]
