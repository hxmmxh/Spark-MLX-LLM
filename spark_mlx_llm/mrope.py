"""NeoX-style three-axis RoPE and still-image position construction."""

import mlx.core as mx
import numpy as np
from mlx import nn


class MultimodalRoPE(nn.Module):
    def __init__(self, dims, base, section, interleaved=False):
        super().__init__()
        self.dims = dims
        pairs = dims // 2
        if interleaved:
            axes = [
                1
                if i % 3 == 1 and i // 3 < section[1]
                else 2
                if i % 3 == 2 and i // 3 < section[2]
                else 0
                for i in range(pairs)
            ]
        else:
            axes = [axis for axis, count in enumerate(section) for _ in range(count)]
        # Derived constants must not become checkpoint parameters.
        self._axes = mx.array(axes)
        self._frequencies = base ** (-mx.arange(pairs, dtype=mx.float32) / pairs)

    def __call__(self, x, positions):
        # x: [batch, heads, length, dim], positions: [3, batch, length].
        angles = positions[self._axes].transpose(1, 2, 0).astype(mx.float32)
        angles = angles * self._frequencies
        cos, sin = mx.cos(angles)[:, None], mx.sin(angles)[:, None]
        half = self.dims // 2
        a, b = (
            x[..., :half].astype(mx.float32),
            x[..., half : self.dims].astype(mx.float32),
        )
        rotated = mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)
        return mx.concatenate([rotated.astype(x.dtype), x[..., self.dims :]], axis=-1)


def build_mrope_positions(length, grids, offsets, merge_size=2):
    """Build [3, 1, length] positions; image offsets are inclusive."""
    if len(grids) != len(offsets):
        raise ValueError("one image grid is required per image offset")
    positions = np.empty((3, length), dtype=np.int32)
    cursor = next_position = 0
    for (t, h, w), (start, end) in zip(grids, offsets):
        if not cursor <= start <= end < length:
            raise ValueError("image offsets overlap or are out of range")
        if t != 1 or min(h, w) <= 0 or h % merge_size or w % merge_size:
            raise ValueError("images require T=1 and grids divisible by merge_size")
        h, w = h // merge_size, w // merge_size
        if h * w != end - start + 1:
            raise ValueError("image grid does not match expanded image token count")
        n = start - cursor
        positions[:, cursor:start] = np.arange(n) + next_position
        next_position += n
        axes = np.indices((t, h, w)).reshape(3, -1)
        positions[:, start : end + 1] = axes + next_position
        next_position += max(t, h, w)
        cursor = end + 1
    positions[:, cursor:] = np.arange(length - cursor) + next_position
    return mx.array(positions[:, None])
