"""Maths shared by training-data conversion and live inference.

Both sides must apply the same rule, otherwise the policy sees different numbers at run time
than it was trained on. Keep them here, not in two copies.
"""

import numpy as np


def normalize(x, lower: float, upper: float, norm_min: float = -1.0, norm_max: float = 1.0):
    """Linearly map ``x`` from [lower, upper] to [norm_min, norm_max].

    A degenerate range (upper <= lower) leaves ``x`` unchanged. ``x`` may be a scalar or an array.
    """
    span = upper - lower
    if span <= 0:
        return x
    return (x - lower) / span * (norm_max - norm_min) + norm_min


def quat_to_euler(q) -> np.ndarray:
    """Quaternion(s) (..., 4) as (x, y, z, w) -> (..., 3) roll, pitch, yaw in radians (extrinsic xyz)."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)
