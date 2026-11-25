"""MLX Spark3 ViT, matching SglangSparkPlugin's vit_4b_hf parameter layout."""

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import BaseModelArgs


@dataclass
class VisionArgs(BaseModelArgs):
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 576
    base_grid: tuple[int, int] | None = None
    out_hidden_size: int = 4096
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    rope_alpha: float = 1.0
    block_backend: str = "native"

    def __post_init__(self):
        if self.block_backend not in ("native", "segmented"):
            raise ValueError("vision block_backend must be native or segmented")
        if self.base_grid is None:
            side = math.isqrt(self.num_position_embeddings)
            self.base_grid = (side, side)
        dims = (
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.patch_size,
            self.temporal_patch_size,
            self.out_hidden_size,
            self.num_position_embeddings,
            *self.base_grid,
        )
        if len(self.base_grid) != 2 or any(type(n) is not int or n <= 0 for n in dims):
            raise ValueError("vision dimensions must be positive integers")
        if math.prod(self.base_grid) != self.num_position_embeddings:
            raise ValueError("vision base_grid must match num_position_embeddings")
        if (
            self.hidden_size % self.num_heads
            or (self.hidden_size // self.num_heads) % 4
        ):
            raise ValueError("vision head dimension must be divisible by four")
        if self.in_channels != 3 or self.spatial_merge_size != 2:
            raise ValueError("vision requires RGB and spatial_merge_size=2")
        if self.hidden_act != "gelu_pytorch_tanh":
            raise ValueError("vision requires hidden_act='gelu_pytorch_tanh'")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.layer_norm_eps, self.rope_theta, self.rope_alpha)
        ):
            raise ValueError("vision epsilon and RoPE parameters must be positive")


def group_spatial(x, height, width):
    return (
        x.reshape(height // 2, 2, width // 2, 2, -1)
        .transpose(0, 2, 1, 3, 4)
        .reshape(height * width, -1)
    )


def _cubic_weights(source, target):
    coordinates = (mx.arange(target, dtype=mx.float32) + 0.5) * source / target - 0.5
    indices = mx.floor(coordinates).astype(mx.int32)[:, None] + mx.arange(-1, 3)
    distance = mx.abs(coordinates[:, None] - indices)
    # PyTorch interpolate(bicubic, align_corners=False): cubic coefficient -0.75.
    a = -0.75
    weights = mx.where(
        distance <= 1,
        ((a + 2) * distance - (a + 3)) * distance**2 + 1,
        mx.where(
            distance < 2,
            ((a * distance - 5 * a) * distance + 8 * a) * distance - 4 * a,
            0,
        ),
    )
    return mx.clip(indices, 0, source - 1), weights


def resize_position_table(table, base_grid, height, width):
    if tuple(base_grid) == (height, width):
        return table[0]
    bh, bw = base_grid
    values = table.reshape(bh, bw, -1).astype(mx.float32)
    rows, rw = _cubic_weights(bh, height)
    columns, cw = _cubic_weights(bw, width)
    values = (mx.take(values, rows, axis=0) * rw[:, :, None, None]).sum(axis=1)
    values = (mx.take(values, columns, axis=1) * cw[None, :, :, None]).sum(axis=2)
    return values.reshape(height * width, -1)


class PatchEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_size = args.patch_size
        self.temporal_patch_size = args.temporal_patch_size
        self.proj = nn.Linear(3 * args.patch_size**2, args.hidden_size)

    def __call__(self, pixels):
        pixels = pixels.reshape(
            -1, 3, self.temporal_patch_size, self.patch_size, self.patch_size
        ).mean(2)
        return self.proj(
            pixels.reshape(pixels.shape[0], -1).astype(self.proj.weight.dtype)
        )


class PositionEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.base_grid = args.base_grid
        self.weight = (
            mx.random.normal((1, args.num_position_embeddings, args.hidden_size))
            * args.hidden_size**-0.5
        )

    def __call__(self, height, width):
        return resize_position_table(self.weight, self.base_grid, height, width)


def apply_axial_rope(x, cos, sin):
    even, odd = x[..., 0::2], x[..., 1::2]
    return mx.stack([even * cos - odd * sin, even * sin + odd * cos], axis=-1).reshape(
        x.shape
    )


class VisionAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.heads = args.num_heads
        self.qkv = nn.Linear(args.hidden_size, 3 * args.hidden_size)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)

    def __call__(self, hidden, cos, sin, lengths):
        q, k, v = mx.split(
            self.qkv(hidden).reshape(len(hidden), 3, self.heads, -1), 3, axis=1
        )
        q, k, v = (x[:, 0] for x in (q, k, v))
        q, k = apply_axial_rope(q, cos, sin), apply_axial_rope(k, cos, sin)
        outputs, start = [], 0
        for length in lengths:
            end = start + length
            qs, ks, vs = (
                mx.contiguous(x[start:end].transpose(1, 0, 2)[None]) for x in (q, k, v)
            )
            output = mx.fast.scaled_dot_product_attention(
                qs, ks, vs, scale=q.shape[-1] ** -0.5
            )
            outputs.append(output[0].transpose(1, 0, 2).reshape(length, -1))
            start = end
        return self.proj(mx.concatenate(outputs))


class VisionMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_fc1 = nn.Linear(args.hidden_size, args.intermediate_size)
        self.linear_fc2 = nn.Linear(args.intermediate_size, args.hidden_size)

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block_backend = args.block_backend
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.attn = VisionAttention(args)
        self.mlp = VisionMLP(args)

    def __call__(self, hidden, cos, sin, lengths):
        if self.block_backend == "segmented":
            from .vit_fusion import forward_block, supports

            if supports(self, hidden, cos, sin, lengths):
                return forward_block(self, hidden, cos, sin, lengths)
        hidden = hidden + self.attn(self.norm1(hidden), cos, sin, lengths)
        return hidden + self.mlp(self.norm2(hidden))


class SpatialMerger(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_eps)
        self.linear_fc1 = nn.Linear(4 * args.hidden_size, 4 * args.hidden_size)
        self.linear_fc2 = nn.Linear(4 * args.hidden_size, args.out_hidden_size)

    def __call__(self, x):
        x = self.norm(x).reshape(-1, self.linear_fc1.weight.shape[1])
        return self.linear_fc2(nn.gelu(self.linear_fc1(x)))


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.patch_embed = PatchEmbedding(args)
        self.pos_embed = PositionEmbedding(args)
        self.blocks = [VisionBlock(args) for _ in range(args.depth)]
        self.merger = SpatialMerger(args)

    def __call__(self, pixel_values, image_grid_thw):
        grids = (
            image_grid_thw.tolist()
            if hasattr(image_grid_thw, "tolist")
            else image_grid_thw
        )
        dim = 3 * self.args.temporal_patch_size * self.args.patch_size**2
        if (
            pixel_values.ndim != 2
            or pixel_values.shape[1] != dim
            or not mx.issubdtype(pixel_values.dtype, mx.floating)
        ):
            raise ValueError(f"vision pixel_values must be floating [N,{dim}]")
        if not grids or any(
            len(g) != 3
            or any(type(n) is not int for n in g)
            or g[0] != 1
            or min(g[1:]) <= 0
            or g[1] % 2
            or g[2] % 2
            for g in grids
        ):
            raise ValueError(
                "vision requires still images (T=1) and positive even H/W grids"
            )
        if sum(h * w for _, h, w in grids) != pixel_values.shape[0]:
            raise ValueError("vision grid patch count differs from pixel_values")
        hidden = self.patch_embed(pixel_values)
        positions, coordinates = [], []
        for _, height, width in grids:
            positions.append(
                group_spatial(self.pos_embed(height, width), height, width)
            )
            rows, columns = mx.meshgrid(
                mx.arange(height), mx.arange(width), indexing="ij"
            )
            coordinates.append(
                group_spatial(mx.stack([rows, columns], axis=-1), height, width)
            )
        hidden = hidden + mx.concatenate(positions).astype(hidden.dtype)
        axis_dim = self.args.hidden_size // self.args.num_heads // 2
        frequency = self.args.rope_theta ** (
            -mx.arange(0, axis_dim, 2, dtype=mx.float32) / axis_dim
        )
        angle = (
            mx.concatenate(coordinates)[..., None] * frequency * self.args.rope_alpha
        ).reshape(len(hidden), 1, -1)
        cos, sin = (
            mx.cos(angle).astype(hidden.dtype),
            mx.sin(angle).astype(hidden.dtype),
        )
        lengths = [h * w for _, h, w in grids]
        for block in self.blocks:
            hidden = block(hidden, cos, sin, lengths)
        return self.merger(hidden)
