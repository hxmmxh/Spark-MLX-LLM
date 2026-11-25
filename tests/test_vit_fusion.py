"""固定输入检查最终 ViT 路由和融合结果。"""

import mlx.core as mx
import numpy as np
import pytest

from spark_mlx_llm.spark3_vision import VisionArgs, VisionBlock
from spark_mlx_llm.vit_fusion import forward_block, supports


@pytest.fixture(scope="module")
def block():
    mx.random.seed(241124)
    block = VisionBlock(VisionArgs(depth=1))
    block.set_dtype(mx.bfloat16)
    mx.eval(block.parameters())
    return block


def inputs(tokens, dtype=mx.bfloat16):
    mx.random.seed(241124 + tokens)
    x = mx.random.normal((tokens, 1152)).astype(dtype)
    angle = mx.random.normal((tokens, 1, 36))
    cos, sin = mx.cos(angle).astype(dtype), mx.sin(angle).astype(dtype)
    mx.eval(x, cos, sin)
    return x, cos, sin


def test_backend_validation():
    assert VisionArgs().block_backend == "native"
    assert VisionArgs(block_backend="segmented").block_backend == "segmented"
    with pytest.raises(ValueError, match="block_backend"):
        VisionArgs(block_backend="invalid")


def test_native_fallback(block):
    x, cos, sin = inputs(8, mx.float32)
    block.block_backend = "native"
    expected = block(x, cos, sin, [8])
    block.block_backend = "segmented"
    assert not supports(block, x, cos, sin, [8])
    actual = block(x, cos, sin, [8])
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
    block.block_backend = "native"


@pytest.mark.parametrize("tokens", [196, 1040, 2304, 3844])
def test_segmented_block_matches_native(block, tokens):
    x, cos, sin = inputs(tokens)
    if not supports(block, x, cos, sin, [tokens]):
        pytest.skip("Requires MLX 0.32.4 and Apple M5 Pro")
    # This also checks the split-K and regular FC2 routes, and both SDPA routes.
    from spark_mlx_llm.benchmark.runtime import DeviceLock

    with DeviceLock().acquire(timeout=45):
        block.block_backend = "native"
        expected = block(x, cos, sin, [tokens])
        result = forward_block(block, x, cos, sin, [tokens])
        block.block_backend = "segmented"
        dispatched = block(x, cos, sin, [tokens])
        mx.eval(expected, result, dispatched)
        mx.synchronize()
        assert mx.array_equal(expected.view(mx.uint16), result.view(mx.uint16)).item()
        assert mx.array_equal(
            expected.view(mx.uint16), dispatched.view(mx.uint16)
        ).item()
        block.block_backend = "native"


def test_multi_image_keeps_native_path(block):
    x, cos, sin = inputs(196)
    assert not supports(block, x, cos, sin, [98, 98])
    block.block_backend = "native"
    expected = block(x, cos, sin, [98, 98])
    block.block_backend = "segmented"
    actual = block(x, cos, sin, [98, 98])
    mx.eval(expected, actual)
    assert mx.array_equal(expected.view(mx.uint16), actual.view(mx.uint16)).item()
    block.block_backend = "native"
