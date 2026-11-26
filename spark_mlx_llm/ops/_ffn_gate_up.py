"""Shared projection packing for the Spark3 W4/G32 gate and up projections."""

from threading import Lock

import mlx.core as mx
from mlx import nn


def _parameters(module):
    if type(module) is nn.Linear:
        return (module.weight,)
    return (module.weight, module.scales, module.biases,
            module.group_size, module.bits)


def _project(x, parameters):
    if len(parameters) == 1:
        return x @ parameters[0].T
    weight, scales, biases, group_size, bits = parameters
    return mx.quantized_matmul(x, weight, scales=scales, biases=biases,
                              transpose=True, group_size=group_size, bits=bits)


def _pack_alias(modules):
    """打包后以原形状视图替换参数，原始和融合路径共用同一份数据。"""
    params = [_parameters(module) for module in modules]
    if len({len(p) for p in params}) != 1:
        return None
    if len(params[0]) > 1 and len({p[3:] for p in params}) != 1:
        return None
    if len(modules) == 1:
        return params[0]
    arrays = 1 if len(params[0]) == 1 else 3
    packed = tuple(mx.concatenate([p[i] for p in params], axis=0)
                   for i in range(arrays))
    mx.eval(*packed)
    start = 0
    names = ('weight',) if arrays == 1 else ('weight', 'scales', 'biases')
    for module, original in zip(modules, params):
        end = start + original[0].shape[0]
        for name, value in zip(names, packed):
            setattr(module, name, value[start:end])
        start = end
    return packed + params[0][arrays:]


_PACKED_FFN_ROWS = (1, 4, 8)
_PACK_GATE_UP_LOCK = Lock()


def _packed_w4_gate_up(mlp):
    """Reuse one W4/G32 gate/up buffer across native and segmented FFNs."""
    gate, up = mlp.gate_proj, mlp.up_proj
    if not all(
        type(proj) is nn.QuantizedLinear
        and (proj.bits, proj.group_size) == (4, 32)
        and proj.mode == 'affine' and 'bias' not in proj
        and proj.biases is not None
        and proj.scales.dtype == proj.biases.dtype == mx.bfloat16
        for proj in (gate, up)
    ) or gate.weight.shape != up.weight.shape:
        return None

    def sources():
        return gate.weight, gate.scales, gate.biases, up.weight, up.scales, up.biases

    def get_cached(current):
        cached = getattr(mlp, '_packed_gate_up', None)
        if cached is not None and all(a is b for a, b in zip(cached[0], current)):
            return cached[1]
        return None

    packed = get_cached(sources())
    if packed is not None:
        return packed
    with _PACK_GATE_UP_LOCK:
        packed = get_cached(sources())
        if packed is None:
            packed = _pack_alias((gate, up))
            object.__setattr__(mlp, '_packed_gate_up', (sources(), packed))
        return packed


@mx.compile
def _packed_ffn(z, h, gate_up, down):
    gate, up = mx.split(_project(z, gate_up), 2, axis=-1)
    return h + _project(nn.gelu(gate) * up, down)
