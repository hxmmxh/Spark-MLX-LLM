"""Experimental and loading-time draft quantization backends.

This changes the draft's numeric proposal distribution. Acceptance must always
use the proposal probabilities actually returned by that new draft. Experimental
handles retain original modules for restoration. Runtime profiles keep only
quantized projections and preserve checkpoint files and target weights.
"""

import math

import mlx.core as mx
from mlx import nn

from ..extension import get_extension


class _DenseReadout(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = weight
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        value = x @ self.weight.T
        return value + self.bias if "bias" in self else value


class DerivedDraftPerChannelW8Linear(nn.Module):
    """Derived symmetric INT8 proposal weights with FP32 per-channel scales.

    Eligible GPU M2..8 calls use the explicit public W8 multirow candidate;
    one-row calls use the existing W8A16 primitive. Unsupported shapes/devices
    retain the supplied original dense BF16 projection. This is a changed,
    shape-dependent proposal backend and is deliberately opt-in.
    """

    def __init__(self, weight, *, bias=None, fallback=None, variant="nax"):
        super().__init__()
        if (
            not isinstance(weight, mx.array)
            or weight.ndim != 2
            or weight.dtype != mx.bfloat16
            or min(weight.shape) <= 0
        ):
            raise ValueError(
                "Derived per-channel W8 requires a positive BF16 weight matrix"
            )
        if variant not in ("nax", "matrix", "wide"):
            raise ValueError("Unknown per-channel W8 multirow variant")
        scores = weight.astype(mx.float32)
        scale = mx.max(mx.abs(scores), axis=-1)
        self.scales = mx.where(scale > 0, scale / 127, 1)
        self.weight = mx.clip(
            mx.round(scores / self.scales[:, None]), -127, 127
        ).astype(mx.int8)
        if bias is not None:
            self.bias = bias
        object.__setattr__(
            self,
            "_fallback",
            fallback if fallback is not None else _DenseReadout(weight, bias),
        )
        self._variant = variant
        self.freeze()

    def __call__(self, x):
        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and x.dtype == mx.bfloat16
            and x.shape[-1] % 32 == 0
        ):
            ext = get_extension()
            if ext is None:
                return self._fallback(x)
            rows = math.prod(x.shape[:-1])
            if rows == 1:
                operation = getattr(ext, "w8a16_per_channel", None)
                if not callable(operation):
                    return self._fallback(x)
                value = operation(x, self.weight, self.scales)
            elif 2 <= rows <= 8 and (
                self._variant != "nax"
                or self.weight.shape[0] % 32 == 0
                and x.shape[-1] % 64 == 0
            ):
                operation = getattr(ext, "w8a16_multirow", None)
                available = getattr(ext, "is_w8a8_available", None)
                if (
                    not callable(operation)
                    or self._variant == "nax"
                    and (not callable(available) or not available())
                ):
                    return self._fallback(x)
                value = operation(
                    x,
                    self.weight,
                    self.scales,
                    output_dtype=mx.bfloat16,
                    variant=self._variant,
                    simdgroups=4,
                )
            else:
                return self._fallback(x)
            return value + self.bias if "bias" in self else value
        return self._fallback(x)


class DerivedDraftW8Linear(nn.QuantizedLinear):
    """Native affine W8 layer made directly from supplied BF16 checkpoint values.

    Calling the base QuantizedLinear constructor would allocate random temporary
    weights and advance RNG state. Initializing Module and assigning all public
    quantization fields avoids both, then uses the official inherited forward.
    """

    def __init__(self, weight, *, bias=None, group_size=32):
        nn.Module.__init__(self)
        if (
            not isinstance(weight, mx.array)
            or weight.ndim != 2
            or weight.dtype != mx.bfloat16
            or min(weight.shape) <= 0
        ):
            raise ValueError("Derived draft W8 requires a BF16 weight matrix")
        if (
            type(group_size) is not int
            or group_size not in (32, 64, 128)
            or weight.shape[1] % group_size
        ):
            raise ValueError("Derived draft W8 requires group_size32/64/128 dividing K")
        self.group_size, self.bits, self.mode = group_size, 8, "affine"
        self.weight, self.scales, self.biases = mx.quantize(
            weight, group_size=group_size, bits=8, mode="affine"
        )
        if bias is not None:
            if (
                not isinstance(bias, mx.array)
                or bias.shape != (weight.shape[0],)
                or bias.dtype != mx.bfloat16
            ):
                raise ValueError("Derived draft W8 bias must be BF16[N]")
            self.bias = bias
        self.freeze()


def _parent(root, path):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    return parent, parts[-1]


class DerivedDraftW8:
    """Restorable experiment backend built from original BF16 draft weights.

    Only exact dense nn.Linear modules are replaced. Embeddings, norms and the
    target are untouched. Markov and external readout changes are explicit.
    No checkpoint file is written and no initialization consumes random draws.
    """

    def __init__(
        self,
        drafter,
        *,
        group_size=32,
        quantize_body=True,
        include_markov=False,
        readout_weight=None,
        readout_projection=None,
        per_channel=False,
        operation_variant="nax",
    ):
        from .drafter import DSparkDrafter

        if type(drafter) is not DSparkDrafter:
            raise TypeError("Derived draft backend requires exact DSparkDrafter")
        if readout_weight is not None and readout_projection is not None:
            raise ValueError(
                "Choose derived readout_weight or supplied readout_projection"
            )
        if (
            readout_weight is not None or readout_projection is not None
        ) and drafter._readout_mode != "external":
            raise ValueError(
                "Derived external readout requires the external readout mode"
            )
        self._drafter = drafter
        self._replacements = []
        self._source_weights = []
        self._original_readout = drafter._external_lm_head
        self._new_readout = None
        self._restored = False
        self.description = {
            "proposal_backend": "derived_symmetric_per_channel_w8"
            if per_channel
            else "derived_affine_w8",
            "group_size": None if per_channel else group_size,
            "bits": 8,
            "include_markov": include_markov,
            "quantize_body": quantize_body,
            "readout": "unchanged",
            "changed_names": [],
            "checkpoint_modified": False,
            "target_modified": False,
            "operation_variant": operation_variant if per_channel else "native_affine",
            "dense_fallback": "original_BF16_for_unsupported_device_or_shape"
            if per_channel
            else None,
        }
        try:
            for name, module in list(drafter.named_modules()):
                if (
                    type(module) is not nn.Linear
                    or name.startswith("confidence_head.")
                    or name == "lm_head"
                ):
                    continue
                if name.startswith("markov_head.") and not include_markov:
                    continue
                if not name.startswith("markov_head.") and not quantize_body:
                    continue
                if module.weight.dtype != mx.bfloat16:
                    raise ValueError(f"{name}: expected BF16 source weight")
                derived = (
                    DerivedDraftPerChannelW8Linear(
                        module.weight,
                        bias=module.get("bias"),
                        fallback=module,
                        variant=operation_variant,
                    )
                    if per_channel
                    else DerivedDraftW8Linear(
                        module.weight, bias=module.get("bias"), group_size=group_size
                    )
                )
                mx.eval(derived.parameters())
                parent, key = _parent(drafter, name)
                self._replacements.append((parent, key, module, derived))
                self._source_weights.append((module, module.weight, module.get("bias")))
                parent[key] = derived
                self.description["changed_names"].append(name)
            if readout_weight is not None:
                if readout_weight.shape != (
                    drafter.config.draft_vocab_size,
                    drafter.config.hidden_size,
                ):
                    raise ValueError(
                        "Derived draft readout weight differs from vocabulary/hidden geometry"
                    )
                self._new_readout = (
                    DerivedDraftPerChannelW8Linear(
                        readout_weight, variant=operation_variant
                    )
                    if per_channel
                    else DerivedDraftW8Linear(readout_weight, group_size=group_size)
                )
                mx.eval(self._new_readout.parameters())
                self.description["readout"] = (
                    "derived_original_bf16_weight_per_channel_w8"
                    if per_channel
                    else "derived_original_bf16_weight_w8"
                )
            elif readout_projection is not None:
                if not callable(readout_projection):
                    raise TypeError("readout_projection must be callable")
                self._new_readout = readout_projection
                self.description["readout"] = "supplied_projection_changes_proposal"
            if self._new_readout is not None:
                drafter.bind_lm_head(self._new_readout)
        except BaseException:
            for parent, key, original, _ in reversed(self._replacements):
                parent[key] = original
            if drafter._readout_mode == "external":
                object.__setattr__(drafter, "_external_lm_head", self._original_readout)
            raise

    def activate(self):
        """Reinstall precomputed proposal tensors without quantization or RNG work."""
        if not self._restored:
            if any(
                parent[key] is not derived
                for parent, key, _, derived in self._replacements
            ):
                raise RuntimeError("Active derived draft projection changed")
            if (
                self._new_readout is not None
                and self._drafter._external_lm_head is not self._new_readout
            ):
                raise RuntimeError("Active derived draft readout changed")
            return self
        if any(
            module.weight is not weight or module.get("bias") is not bias
            for module, weight, bias in self._source_weights
        ):
            raise RuntimeError(
                "Source draft weights changed; prepare a fresh derived backend"
            )
        if any(
            parent[key] is not original
            for parent, key, original, _ in self._replacements
        ):
            raise RuntimeError(
                "Restore other draft projections before activating this backend"
            )
        if (
            self._new_readout is not None
            and self._drafter._external_lm_head is not self._original_readout
        ):
            raise RuntimeError(
                "Restore original draft readout before activating this backend"
            )
        for parent, key, _, derived in self._replacements:
            parent[key] = derived
        if self._new_readout is not None:
            self._drafter.bind_lm_head(self._new_readout)
        self._restored = False
        return self

    def restore(self):
        if self._restored:
            return
        if self._new_readout is not None:
            if self._drafter._external_lm_head is not self._new_readout:
                raise RuntimeError(
                    "Draft readout changed before derived backend restore"
                )
            object.__setattr__(
                self._drafter, "_external_lm_head", self._original_readout
            )
        for parent, key, original, derived in reversed(self._replacements):
            if parent[key] is not derived:
                raise RuntimeError(
                    "Draft projection changed before derived backend restore"
                )
            parent[key] = original
        self._restored = True

    def __call__(self):
        self.restore()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.restore()


def install_derived_draft_w8(drafter, **kwargs):
    """Prepare W8 proposal tensors outside timing and return a restore handle."""
    return DerivedDraftW8(drafter, **kwargs)


class _QuantizedPerChannelReadout(nn.Module):
    """Portable execution of the stored INT8 codes and FP32 channel scales."""

    def __init__(self, weight, scales, bias=None):
        super().__init__()
        self.weight, self.scales = weight, scales
        if bias is not None:
            self.bias = bias

    def __call__(self, x):
        value = (
            (x.astype(mx.float32) @ self.weight.astype(mx.float32).T) * self.scales
        ).astype(x.dtype)
        return value + self.bias if "bias" in self else value


class RuntimeDraftW8Linear(DerivedDraftPerChannelW8Linear):
    """Deployable W8 with a quantized fallback and no retained source module.

    FP32 Metal calls with 1..8 rows use direct INT8 reads and FP32 reduction,
    without expanding the weight matrix. BF16 calls retain the existing W8
    kernels. Other devices, shapes or unavailable extensions compute from the
    same quantized weights in floating point; no source BF16 matrix is kept.
    """

    def __init__(self, weight, *, bias=None):
        # The source is floating point, including FP32 checkpoint tensors or an
        # explicit dtype override. Quantization arithmetic uses FP32 throughout.
        nn.Module.__init__(self)
        if weight.ndim != 2 or not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError("Runtime W8 requires a floating-point matrix")
        scores = weight.astype(mx.float32)
        maximum = mx.max(mx.abs(scores), axis=-1)
        self.scales = mx.where(maximum > 0, maximum / 127, 1)
        self.weight = mx.clip(
            mx.round(scores / self.scales[:, None]), -127, 127
        ).astype(mx.int8)
        if bias is not None:
            self.bias = bias
        self._finish()

    @classmethod
    def from_quantized(cls, weight, scales, *, bias=None):
        """Restore serialized codes without quantization or random initialization."""
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.weight, layer.scales = weight, scales
        if bias is not None:
            layer.bias = bias
        layer._finish()
        return layer

    def _finish(self):
        self._variant = "nax"
        object.__setattr__(
            self,
            "_fallback",
            _QuantizedPerChannelReadout(self.weight, self.scales, self.get("bias")),
        )
        # Exact module type checks in optimized sampling/context paths also
        # consult this capability flag before invoking extension kernels.
        enabled = False
        if mx.default_device() == mx.gpu and mx.metal.is_available():
            ext = get_extension()
            enabled = bool(getattr(ext, "is_w8a8_available", lambda: False)())
        object.__setattr__(self, "_native_kernels", enabled)
        self.freeze()

    def __call__(self, x):
        if (
            isinstance(x, mx.array)
            and x.dtype == mx.float32
            and x.ndim >= 1
            and 1 <= math.prod(x.shape[:-1]) <= 8
            and self.weight.ndim == 2
            and min(self.weight.shape) > 0
            and self.weight.dtype == mx.int8
            and self.weight.shape[1] == x.shape[-1]
            and self.scales.dtype == mx.float32
            and self.scales.shape == (self.weight.shape[0],)
            and (
                "bias" not in self
                or isinstance(self.bias, mx.array)
                and self.bias.shape == (self.weight.shape[0],)
                and self.bias.dtype in (mx.float16, mx.bfloat16, mx.float32)
            )
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        ):
            # Independent of _native_kernels, which guards the BF16/NAX path.
            # Preserve the module and its stored codes/scales for serialization
            # and the exact-type checks in compiled Markov sampling.
            operation = getattr(get_extension(), "w8a32_per_channel", None)
            if callable(operation):
                return operation(x, self.weight, self.scales, bias=self.get("bias"))
        if not self._native_kernels:
            return self._fallback(x)
        return super().__call__(x)


class RuntimeDraftW4Linear(nn.QuantizedLinear):
    """Target-rule asymmetric W4/G32 with BF16 scales and affine offsets."""

    def __init__(self, weight, *, bias=None, chunk_rows=256):
        import numpy as np

        from ..conversion import _coefficient, _packed

        if (
            weight.ndim != 2
            or min(weight.shape) <= 0
            or weight.shape[1] % 32
            or not mx.issubdtype(weight.dtype, mx.floating)
        ):
            raise ValueError(
                "Runtime INT4/G32 requires a floating matrix with K divisible by 32"
            )
        nn.Module.__init__(self)
        self.bits, self.group_size, self.mode = 4, 32, "affine"
        packed, scales, biases = [], [], []
        for first in range(0, weight.shape[0], chunk_rows):
            values = np.array(weight[first : first + chunk_rows].astype(mx.float32))
            if not np.isfinite(values).all():
                raise ValueError("Non-finite runtime INT4 source weights")
            grouped = values.reshape(values.shape[0], -1, 32)
            lower = np.minimum(grouped.min(axis=-1), 0)
            upper = np.maximum(grouped.max(axis=-1), 0)
            step = (upper - lower) / np.float32(15)
            step = np.where(step == 0, np.float32(1), step)
            scale_bits, rounded = _coefficient(step, "BF16")
            if not (rounded > 0).all():
                raise ValueError("Runtime INT4 scale underflow")
            zeros = np.clip(np.rint(-lower / rounded), 0, 15)
            codes = np.clip(
                np.rint(grouped / rounded[..., None]) + zeros[..., None], 0, 15
            )
            bias_bits, _ = _coefficient(-zeros * rounded, "BF16")
            packed.append(mx.array(_packed(codes.reshape(values.shape), 4)))
            scales.append(mx.array(scale_bits).view(mx.bfloat16))
            biases.append(mx.array(bias_bits).view(mx.bfloat16))
        self.weight = mx.concatenate(packed)
        self.scales = mx.concatenate(scales)
        self.biases = mx.concatenate(biases)
        if bias is not None:
            self.bias = bias
        self.freeze()

    @classmethod
    def from_quantized(cls, weight, scales, biases, *, bias=None):
        """Restore the packed affine representation exactly as saved."""
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        layer.bits, layer.group_size, layer.mode = 4, 32, "affine"
        layer.weight, layer.scales, layer.biases = weight, scales, biases
        if bias is not None:
            layer.bias = bias
        layer.freeze()
        return layer


def restore_runtime_draft(drafter, weights, schema):
    """Install validated serialized projections before strict parameter loading."""
    report = {
        "policy": schema["policy"],
        "loaded_from_checkpoint": True,
        "int4_modules": [],
        "int8_modules": [],
        "int4_group_size": 32 if schema["policy"] == "int4" else None,
        "int8_scheme": "symmetric_per_channel_fp32_scales",
        "external_readout": "unchanged",
    }
    for name, spec in schema["modules"].items():
        args = [weights[name + ".weight"], weights[name + ".scales"]]
        if spec["bits"] == 4:
            args.append(weights[name + ".biases"])
        constructor = (
            RuntimeDraftW4Linear if spec["bits"] == 4 else RuntimeDraftW8Linear
        )
        layer = constructor.from_quantized(*args, bias=weights.get(name + ".bias"))
        parent, key = _parent(drafter, name)
        parent[key] = layer
        report["int4_modules" if spec["bits"] == 4 else "int8_modules"].append(name)
    return report


def quantize_runtime_draft(drafter, policy="bf16"):
    """Quantize selected draft modules once after strict checkpoint loading.

    INT8: W8 per output channel on fc, attention QKV/out, FFN, the Markov
    projection and an independent draft lm_head. INT4: replace QKV/FFN with
    target-rule W4/G32 and retain W8 on the remaining selected modules. Token
    embeddings, attention gates, norms, confidence and external readout stay
    under their existing policies. No restore handle retains source matrices.
    """
    if policy not in ("bf16", "int8", "int4"):
        raise ValueError("draft_quantization must be bf16, int8 or int4")
    report = {
        "policy": policy,
        "int4_group_size": 32 if policy == "int4" else None,
        "int8_scheme": "symmetric_per_channel_fp32_scales"
        if policy != "bf16"
        else None,
        "int4_modules": [],
        "int8_modules": [],
        "kept_float_modules": [],
        "external_readout": "unchanged",
    }
    if policy == "bf16":
        return report
    candidates = []
    for name, module in drafter.named_modules():
        if type(module) is not nn.Linear:
            continue
        if name.startswith("confidence_head.") or name.endswith(".self_attn.g_proj"):
            report["kept_float_modules"].append(name)
            continue
        if not mx.issubdtype(module.weight.dtype, mx.floating):
            raise ValueError(
                f"{name}: loading-time quantization requires floating source weights"
            )
        is_w4 = policy == "int4" and (
            name.endswith(".self_attn.q_k_v_proj")
            or ".mlp." in name
            and name.rsplit(".", 1)[-1] in ("gate_proj", "up_proj", "down_proj")
        )
        if is_w4 and module.weight.shape[1] % 32:
            raise ValueError(f"{name}: INT4/G32 requires K divisible by 32")
        candidates.append((name, module, is_w4))
    for name, original, is_w4 in candidates:
        constructor = RuntimeDraftW4Linear if is_w4 else RuntimeDraftW8Linear
        quantized = constructor(original.weight, bias=original.get("bias"))
        mx.eval(quantized.parameters())
        parent, key = _parent(drafter, name)
        parent[key] = quantized
        report["int4_modules" if is_w4 else "int8_modules"].append(name)
    return report
