"""Spark X3 SWA/FULL inference, matching the Spark3 vLLM plugin checkpoint.

CED consumers have their own queries and reuse the producer's post-RoPE KV.
Only cache owners appear in make_cache(), in ascending layer order, so native
MLX LM cache save/load and trimming never visit the same cache twice.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import mlx.core as mx
from mlx import nn
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_causal_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from .mrope import MultimodalRoPE, build_mrope_positions
from .ops import _swa_prefill
from .ops.sdpa import SDPAConfig, dispatch_sdpa
from .spark3_vision import VisionArgs, VisionTransformer


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "spark3"
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 131072
    sliding_window: int = 512
    q_lora_rank: int | None = None
    kv_source_layer_ids: int | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None
    layer_types: list[str] | None = None
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 8192
    headwise_attn_output_gate: bool = True
    tie_word_embeddings: bool = False
    hidden_act: str = "gelu"
    gate_attn_act_mode: str = "sigmoid"
    attention_bias: bool = False
    mlp_bias: bool = False
    quantization_config: dict[str, Any] | None = None
    ced_prefill_keep_tokens: int | Literal["auto"] | None = "auto"
    apply_q_lora: bool = True
    vision_config: dict | VisionArgs | None = None
    image_token_id: int | None = None
    image_token: str | None = None
    swa_prefill_kv_skip: bool = True
    fused_quantized_sdpa: bool = True

    @classmethod
    def from_dict(cls, params):
        # HF multimodal exports nest text fields; explicit root fields win.
        fields = dict(params.get("text_config") or {})
        fields.update({k: v for k, v in params.items() if k != "text_config"})
        if any(
            fields.get(k) is not None
            for k in (
                "indexer_n_heads",
                "indexer_kv_heads",
                "indexer_head_dim",
                "indexer_budget",
                "indexer_compress_ratio",
            )
        ):
            raise ValueError("Spark3 QSA is not supported")
        if fields.get("rope_parameters") is None and fields.get("rope_scaling"):
            fields["rope_parameters"] = fields["rope_scaling"]
        return super().from_dict(fields)

    def __post_init__(self):
        SDPAConfig.validate_values(vars(self))
        if self.model_type != "spark3":
            raise ValueError(f"Expected model_type 'spark3', got {self.model_type!r}")
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "sliding_window",
            "q_lora_rank",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must match num_hidden_layers")
        unsupported = set(self.layer_types) - {"sliding_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"Unsupported Spark3 layer types: {sorted(unsupported)}")
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_theta": 5_000_000,
                    "partial_rotary_factor": 0.25,
                },
                "sliding_attention": {
                    "rope_theta": 10_000,
                    "partial_rotary_factor": 1.0,
                },
            }
        self.rope_parameters = deepcopy(self.rope_parameters)
        if not any(
            k in self.rope_parameters for k in ("full_attention", "sliding_attention")
        ):
            self.rope_parameters = {
                kind: deepcopy(self.rope_parameters) for kind in set(self.layer_types)
            }
        for layer_type in set(self.layer_types):
            if layer_type not in self.rope_parameters:
                raise ValueError(f"Missing RoPE parameters for {layer_type}")
            params = self.rope_parameters[layer_type]
            dims = int(self.head_dim * params.get("partial_rotary_factor", 1.0))
            if dims <= 0 or dims > self.head_dim or dims % 2:
                raise ValueError(f"Invalid rotary dimension {dims} for {layer_type}")
            theta = params.get("rope_theta", 10000.0)
            if (
                not isinstance(theta, (int, float))
                or not math.isfinite(theta)
                or theta <= 0
            ):
                raise ValueError("rope_theta must be finite and positive")
            rope_type = params.get("rope_type", params.get("type", "default"))
            if rope_type not in ("default", "mrope"):
                raise ValueError("Spark3 supports default RoPE and mRoPE only")
            if type(params.get("mrope_interleaved", False)) is not bool:
                raise ValueError("mrope_interleaved must be a boolean")
            section = params.get("mrope_section")
            if "mrope_section" in params:
                if (
                    not isinstance(section, (list, tuple))
                    or len(section) != 3
                    or any(type(n) is not int or n <= 0 for n in section)
                    or 2 * sum(section) != dims
                ):
                    raise ValueError(
                        "mrope_section must have three positive sizes summing to rotary_dim/2"
                    )
            elif rope_type == "mrope" or params.get("mrope_interleaved", False):
                raise ValueError("mRoPE requires mrope_section")
        if isinstance(self.kv_source_layer_ids, list):
            if len(self.kv_source_layer_ids) != 1:
                raise ValueError("kv_source_layer_ids requires exactly one producer")
            self.kv_source_layer_ids = self.kv_source_layer_ids[0]
        source = self.kv_source_layer_ids
        if source is not None:
            if (
                not isinstance(source, int)
                or isinstance(source, bool)
                or not 0 <= source < self.num_hidden_layers
            ):
                raise ValueError("kv_source_layer_ids must be a valid zero-based index")
            if self.layer_types[source] != "full_attention":
                raise ValueError("CED producer must be a full_attention layer")
        if self.hidden_act != "gelu" or self.gate_attn_act_mode != "sigmoid":
            raise ValueError("Spark3 requires GELU MLP and sigmoid attention gates")
        if self.attention_bias or self.mlp_bias:
            raise ValueError("Spark3 checkpoint projections must be bias-free")
        if self.apply_q_lora is not True:
            raise ValueError("Spark3 MLX requires apply_q_lora=true")
        if self.vision_config is not None:
            if isinstance(self.vision_config, dict):
                config = dict(self.vision_config)
                config.setdefault("out_hidden_size", self.hidden_size)
                self.vision_config = VisionArgs.from_dict(config)
            if self.vision_config.out_hidden_size != self.hidden_size:
                raise ValueError("vision out_hidden_size must equal text hidden_size")
            if type(self.image_token_id) is not int:
                raise ValueError("vision requires an integer image_token_id")
            if self.image_token_id >= self.vocab_size:
                raise ValueError(
                    "image_token_id must be inside the vocabulary or negative"
                )
        keep = self.ced_prefill_keep_tokens
        if keep == "auto":
            if source is not None and "full_attention" in self.layer_types[source + 1 :]:
                self.ced_prefill_keep_tokens = self.sliding_window
            else:
                self.ced_prefill_keep_tokens = None
        elif keep is not None:
            if type(keep) is not int or keep < self.sliding_window:
                raise ValueError("ced_prefill_keep_tokens must cover the SWA window")
            if source is None or source == self.num_hidden_layers - 1:
                raise ValueError("ced_prefill_keep_tokens requires layers after a CED producer")

    def is_ced_consumer(self, layer_idx: int) -> bool:
        return (
            self.kv_source_layer_ids is not None
            and layer_idx > self.kv_source_layer_ids
            and self.layer_types[layer_idx] == "full_attention"
        )

    @property
    def uses_mrope(self):
        return any("mrope_section" in p for p in self.rope_parameters.values())


def _make_rope(args: ModelArgs, layer_type: str):
    params = args.rope_parameters[layer_type]
    rotary_dims = int(args.head_dim * float(params.get("partial_rotary_factor", 1.0)))
    if "mrope_section" in params:
        return MultimodalRoPE(
            rotary_dims,
            float(params.get("rope_theta", 10000.0)),
            params["mrope_section"],
            params.get("mrope_interleaved", False),
        )
    return nn.RoPE(
        rotary_dims,
        traditional=False,
        base=float(params.get("rope_theta", 10000.0)),
    )


def _validate_w8a8_activations(activations):
    expected = {
        "num_bits": 8,
        "type": "int",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    if (
        not isinstance(activations, dict)
        or activations.get("num_bits") != 8
        or any(activations.get(key, value) != value for key, value in expected.items())
    ):
        raise ValueError(
            "Spark3 W8A8 requires symmetric dynamic per-token INT8 input_activations"
        )


def _out_proj_uses_w8a8(args: ModelArgs, layer_idx: int) -> bool:
    config = args.quantization_config or {}
    scheme = config.get(f"model.layers.{layer_idx}.self_attn.out_proj", {})
    if not isinstance(scheme, dict) or scheme.get("input_activations") is None:
        return False
    _validate_w8a8_activations(scheme["input_activations"])
    if (
        scheme.get("bits", config.get("bits", 4)) != 8
        or scheme.get("mode", config.get("mode", "affine")) != "affine"
    ):
        raise ValueError("Spark3 W8A8 out_proj requires 8-bit affine packed weights")
    return True


def _w8a8_operation():
    from .extension import get_extension

    extension = get_extension()
    if extension is None or not extension.is_w8a8_available():
        return None
    return extension.w8a8_linear


class _W8A8QuantizedLinear(nn.QuantizedLinear):
    """Spark3 affine checkpoint layout with derived native INT8 kernel inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._operation = _w8a8_operation()
        if self._operation is None:
            raise RuntimeError("Spark3 native W8A8 requires a supported extension")
        object.__setattr__(self, "_w8a8_cache", None)

    def _prepare_w8a8(self):
        sources = self.weight, self.scales, self.biases
        cached = self._w8a8_cache
        if cached is not None and all(
            current is previous for current, previous in zip(sources, cached[0])
        ):
            return cached[1:]
        if self.bits != 8 or self.mode != "affine" or self.weight.dtype != mx.uint32:
            raise ValueError("Spark3 W8A8 requires UINT32-packed 8-bit affine weights")
        input_dims = self.weight.shape[-1] * 4
        if input_dims > 131040 or input_dims % 32:
            raise ValueError(
                "Spark3 W8A8 input width must be a multiple of 32 <= 131040"
            )
        scales, biases = self.scales.astype(mx.float32), self.biases.astype(mx.float32)
        valid = (
            mx.all(mx.isfinite(scales) & (scales > 0))
            & mx.all(scales == scales[:, :1])
            & mx.all(biases == -128 * scales)
        )
        if not bool(valid):
            raise ValueError(
                "Spark3 W8A8 requires repeated per-channel scales and biases=-128*scales"
            )
        int8_weights = (self.weight ^ mx.array(0x80808080, dtype=mx.uint32)).view(
            mx.int8
        )
        channel_scales = mx.contiguous(scales[:, 0])
        mx.eval(int8_weights, channel_scales)
        # These derived arrays are not serialized as model parameters.
        object.__setattr__(self, "_w8a8_cache", (sources, int8_weights, channel_scales))
        return int8_weights, channel_scales

    def __call__(self, x):
        weights, scales = self._prepare_w8a8()
        return self._operation(x, weights, scales, self.get("bias"))


class _W8A8OutputProjection(nn.Linear):
    def to_quantized(
        self,
        group_size=None,
        bits=None,
        mode="affine",
        quantize_input=False,
        input_activations=None,
    ):
        if input_activations is not None:
            _validate_w8a8_activations(input_activations)
        if bits != 8 or mode != "affine":
            raise ValueError(
                "Spark3 W8A8 out_proj requires 8-bit affine packed weights"
            )
        quantized_cls = (
            _W8A8QuantizedLinear
            if _w8a8_operation() is not None
            else nn.QuantizedLinear
        )
        return quantized_cls.from_linear(self, group_size, bits, mode)

    def __call__(self, x):
        raise RuntimeError("Load Spark3's configured 8-bit out_proj before inference")


def _is_affine_quantized_linear(layer):
    return (
        type(layer) is nn.QuantizedLinear
        and layer.mode == "affine"
        and layer.biases is not None
        and "bias" not in layer
    )


def _quantized_linear_args(layer):
    return layer.weight, layer.scales, layer.biases, layer.group_size, layer.bits


def _quantized_linear(x, weight, scales, biases, group_size, bits):
    return mx.quantized_matmul(
        x,
        weight,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
    )


@mx.compile
def _fused_attention_output(x, output, gate_weight, out_params):
    output = mx.contiguous(output.transpose(0, 2, 1, 3))
    gates = mx.sigmoid((x @ gate_weight.T).astype(mx.float32)).astype(output.dtype)
    output = output * gates[..., None]
    return _quantized_linear(output.reshape(x.shape[0], x.shape[1], -1), *out_params)


@mx.compile
def _fused_ffn(
    x, attention_output, norm_weight, gate_params, up_params, down_params, eps
):
    residual = x + attention_output
    normalized = mx.fast.rms_norm(residual, norm_weight, eps)
    gate = _quantized_linear(normalized, *gate_params)
    up = _quantized_linear(normalized, *up_params)
    return residual + _quantized_linear(nn.gelu(gate) * up, *down_params)


class _KVState(NamedTuple):
    keys: Any
    values: Any
    offset: Any
    mask: Any
    cache: Any


class _TailPrefillSWACache:
    """Append a bounded prefill suffix without changing the decode cache type."""

    def __init__(self, cache, block_length, skipped):
        self.cache = cache
        self.offset = cache.offset + skipped
        # Match SGLang's bounded SWA read for each scheduled prefill block.
        # When cropping creates a gap, no KV from before the gap can be read.
        self.prefix = min(cache.offset, max(0, cache.max_size - block_length))

    def make_mask(self, length, *, return_array=False, window_size=None):
        if not return_array and self.prefix + length <= window_size:
            return "causal"
        return create_causal_mask(length, self.prefix, window_size=window_size)

    def update_and_fetch(self, keys, values):
        cache = self.cache
        end = self.offset + keys.shape[2]
        if self.prefix:
            prev_keys = cache._temporal_order(cache.keys)[..., -self.prefix :, :]
            prev_values = cache._temporal_order(cache.values)[..., -self.prefix :, :]
            keys = mx.concatenate((prev_keys, keys), axis=2)
            values = mx.concatenate((prev_values, values), axis=2)
        window = cache.max_size
        stored = min(keys.shape[2], window)
        cache.state = (keys[..., -stored:, :], values[..., -stored:, :])
        # A compact, chronological buffer. Even if rows were skipped, the
        # next token uses the absolute position at the end of the block.
        cache.meta_state = tuple(map(str, (0, window, end, stored)))
        return keys, values


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.sdpa_config = SDPAConfig(fused_quantized_sdpa=args.fused_quantized_sdpa)
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv_heads * self.head_dim
        layer_type = args.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        self.window_size = args.sliding_window if self.is_sliding else None
        self.swa_prefill_kv_skip = args.swa_prefill_kv_skip
        self.is_ced_consumer = args.is_ced_consumer(layer_idx)
        self.headwise_attn_output_gate = args.headwise_attn_output_gate

        self.q_down_proj = nn.Linear(args.hidden_size, args.q_lora_rank, bias=False)
        self.q_down_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_up_proj = nn.Linear(args.q_lora_rank, self.q_size, bias=False)
        if not self.is_ced_consumer:
            self.k_proj = nn.Linear(args.hidden_size, self.kv_size, bias=False)
            self.v_proj = nn.Linear(args.hidden_size, self.kv_size, bias=False)
        if self.headwise_attn_output_gate:
            self.g_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        output_projection = (
            _W8A8OutputProjection if _out_proj_uses_w8a8(args, layer_idx) else nn.Linear
        )
        self.out_proj = output_projection(self.q_size, args.hidden_size, bias=False)
        self.rope = _make_rope(args, layer_type)

    def _attention_output(self, x: mx.array, output: mx.array) -> mx.array:
        if (
            mx.metal.is_available()
            and type(self.g_proj) is nn.Linear
            and "bias" not in self.g_proj
            and _is_affine_quantized_linear(self.out_proj)
        ):
            return _fused_attention_output(
                x, output, self.g_proj.weight, _quantized_linear_args(self.out_proj)
            )
        output = mx.contiguous(output.transpose(0, 2, 1, 3))
        gates = mx.sigmoid(self.g_proj(x).astype(mx.float32)).astype(output.dtype)
        output = output * gates[..., None]
        return self.out_proj(output.reshape(x.shape[0], x.shape[1], self.q_size))

    def __call__(
        self,
        x: mx.array,
        cache: Any | None = None,
        shared_kv: _KVState | None = None,
        positions: mx.array | None = None,
    ) -> tuple[mx.array, _KVState]:
        batch, length, _ = x.shape
        queries = self.q_up_proj(self.q_down_layernorm(self.q_down_proj(x)))
        queries = mx.contiguous(
            queries.reshape(batch, length, self.n_heads, self.head_dim).transpose(
                0, 2, 1, 3
            )
        )
        if self.is_ced_consumer:
            if shared_kv is None:
                raise ValueError("CED consumer requires the producer's KV state")
            state = shared_kv
        else:
            offset = cache.offset if cache is not None else 0
            if isinstance(offset, mx.array):
                # Batch caches update their offset array in place. Preserve
                # the pre-append positions for this layer and its consumers.
                offset = mx.array(offset)
            # Compute the mask and query position BEFORE the owner appends KV.
            # Consumers reuse both, so chunked prefill cannot see future tokens
            # or accidentally apply RoPE at the owner's updated cache offset.
            mask = create_attention_mask(x, cache, window_size=self.window_size)
            keys = (
                self.k_proj(x)
                .reshape(batch, length, self.n_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            values = (
                self.v_proj(x)
                .reshape(batch, length, self.n_kv_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            keys = self._apply_rope(mx.contiguous(keys), offset, positions)
            values = mx.contiguous(values)
            if cache is not None:
                keys, values = cache.update_and_fetch(keys, values)
            state = _KVState(keys, values, offset, mask, cache)

        queries = self._apply_rope(queries, state.offset, positions)
        output = self._try_swa_prefill(queries, state)
        if output is None:
            output = self._native_sdpa(queries, state)
        if self.headwise_attn_output_gate:
            output = self._attention_output(x, output)
        else:
            output = self.out_proj(
                output.transpose(0, 2, 1, 3).reshape(batch, length, self.q_size)
            )
        return output, state

    def _native_sdpa(self, queries, state):
        if getattr(state.cache, "bits", None) == 8:
            return dispatch_sdpa(
                queries,
                state.keys,
                state.values,
                cache=state.cache,
                scale=self.scale,
                mask=state.mask,
                config=self.sdpa_config,
            )
        return scaled_dot_product_attention(
            queries,
            state.keys,
            state.values,
            cache=state.cache,
            scale=self.scale,
            mask=state.mask,
        )

    def _try_swa_prefill(self, queries, state):
        if (
            self.swa_prefill_kv_skip and self.window_size == 512 and not self.is_ced_consumer
            and (state.cache is None or (
                type(state.cache) is RotatingKVCache and state.cache.keep == 0
            ))
            and isinstance(state.mask, mx.array) and state.mask.dtype == mx.bool_
        ):
            return _swa_prefill.try_prefill(queries, state.keys, state.values)
        return None

    def _apply_rope(self, x, offset, positions):
        if isinstance(self.rope, MultimodalRoPE):
            return self.rope(x, positions)
        return self.rope(x, offset=offset)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    @property
    def is_sliding(self) -> bool:
        return self.self_attn.is_sliding

    def _ffn(self, x: mx.array, attention_output: mx.array) -> mx.array:
        if (
            mx.metal.is_available()
            and type(self.mlp) is MLP
            and type(self.post_attention_layernorm) is nn.RMSNorm
        ):
            projections = self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj
            if all(_is_affine_quantized_linear(p) for p in projections):
                gate, up, down = projections
                return _fused_ffn(
                    x,
                    attention_output,
                    self.post_attention_layernorm.weight,
                    _quantized_linear_args(gate),
                    _quantized_linear_args(up),
                    _quantized_linear_args(down),
                    self.post_attention_layernorm.eps,
                )
        residual = x + attention_output
        return residual + self.mlp(self.post_attention_layernorm(residual))

    def __call__(self, x, cache=None, shared_kv=None, positions=None):
        attention_output, state = self.self_attn(
            self.input_layernorm(x),
            cache=cache,
            shared_kv=shared_kv,
            positions=positions,
        )
        return self._ffn(x, attention_output), state


class Spark3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.hidden_size)
        if args.vision_config is not None:
            self.visual = VisionTransformer(args.vision_config)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.cache_layer_ids = tuple(
            i for i in range(args.num_hidden_layers) if not args.is_ced_consumer(i)
        )

    def __call__(
        self, inputs, cache=None, input_embeddings=None, positions=None, *,
        prefill_keep_tokens=None, ced_prefix_only=False,
    ):
        expected = len(self.cache_layer_ids) + int(self.args.uses_mrope)
        if cache is None:
            cache = [None] * expected
        elif len(cache) != expected:
            raise ValueError(
                f"Expected {expected} owner cache entries (including mRoPE state), "
                f"got {len(cache)}"
            )
        hidden_states = (
            input_embeddings if input_embeddings is not None else self.embedding(inputs)
        )
        block_length = hidden_states.shape[1]
        skipped = max(0, block_length - prefill_keep_tokens) if prefill_keep_tokens else 0
        if self.args.uses_mrope:
            batch, length = hidden_states.shape[:2]
            offset = cache[0].offset if cache[0] is not None else 0
            position_cache = cache[-1]
            if positions is None:
                delta = position_cache[0] if position_cache is not None else None
                base = mx.array(offset) + (delta if delta is not None else 0)
                base = mx.broadcast_to(base, (batch,))
                positions = mx.broadcast_to(
                    (base[:, None] + mx.arange(length))[None], (3, batch, length)
                )
            elif positions.ndim == 2:
                positions = positions[:, None]
            if positions.shape != (3, batch, length) or not mx.issubdtype(
                positions.dtype, mx.integer
            ):
                raise ValueError("mRoPE positions must be integer [3,batch,length]")
            if position_cache is not None:
                # Store request-local decode offset in a native serializable cache.
                position_cache[0] = (
                    positions.max(axis=(0, 2)) + 1 - (mx.array(offset) + length)
                )
        owner_caches = iter(cache[: len(self.cache_layer_ids)])
        source = self.args.kv_source_layer_ids
        tail_start = source + 1 if prefill_keep_tokens is not None else len(self.layers)
        shared_kv = None
        for i, layer in enumerate(self.layers):
            if i == tail_start and ced_prefix_only:
                return hidden_states
            if i == tail_start and skipped:
                hidden_states = hidden_states[:, skipped:]
                if positions is not None:
                    positions = positions[..., skipped:]
                mask = shared_kv.mask
                if isinstance(mask, mx.array):
                    mask = mask[..., skipped:, :]
                # An implicit causal mask already aligns queries to the last
                # rows of the producer's full key sequence.
                shared_kv = shared_kv._replace(offset=shared_kv.offset + skipped, mask=mask)
            consumer = layer.self_attn.is_ced_consumer
            layer_cache = None if consumer else next(owner_caches)
            if i >= tail_start and layer.is_sliding:
                layer_cache = _TailPrefillSWACache(layer_cache, block_length, skipped)
            hidden_states, state = layer(
                hidden_states,
                cache=layer_cache,
                shared_kv=shared_kv if consumer else None,
                positions=positions,
            )
            if i == source:
                shared_kv = state
        return self.norm(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Spark3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings=None,
        positions=None,
        pixel_values=None,
        image_grid_thw=None,
        image_offsets=None,
    ):
        if pixel_values is not None:
            if input_embeddings is not None:
                raise ValueError("supply pixels or input_embeddings, not both")
            prepared = self.prepare_inputs(
                inputs, pixel_values, image_grid_thw, image_offsets
            )
            input_embeddings = prepared["input_embeddings"]
            if positions is None:
                positions = prepared["positions"]
                if cache is not None:
                    base = mx.array(cache[0].offset)
                    if self.args.uses_mrope and cache[-1][0] is not None:
                        base = base + cache[-1][0]
                    positions = positions + base.reshape(1, -1, 1)
        elif image_grid_thw is not None or image_offsets is not None:
            raise ValueError("image grids and offsets require pixel_values")
        hidden_states = self.model(inputs, cache, input_embeddings, positions)
        if self.args.tie_word_embeddings:
            return self.model.embedding.as_linear(hidden_states)
        return self.lm_head(hidden_states)

    def prefill_skip_tail_before(self, prompt_tokens, chunk_size):
        """Return the first bulk block that must build late-layer SWA KV."""
        window = self.args.sliding_window
        bulk_tokens = prompt_tokens - 1
        if (
            self.args.ced_prefill_keep_tokens is None
            or chunk_size < max(2, window)
            or bulk_tokens <= chunk_size
        ):
            return 0
        last_start = (bulk_tokens - 1) // chunk_size * chunk_size
        if bulk_tokens - last_start < window:
            # A short last block can read the immediately preceding SWA window.
            last_start -= chunk_size
        return last_start

    def prefill_cache(self, inputs, cache, *, prefix_only=False):
        """Fill an ordinary single-request cache when prompt logits are unused."""
        keep = self.args.ced_prefill_keep_tokens
        if keep is None or inputs.shape[0] != 1 or inputs.shape[1] == 1:
            self(inputs, cache=cache)
            return
        self.model(inputs, cache=cache, prefill_keep_tokens=keep, ced_prefix_only=prefix_only)
        if prefix_only:
            early_count = self.args.kv_source_layer_ids + 1
            owner_count = len(self.model.cache_layer_ids)
            # The retained bulk block recreates the late buffers. Advance their
            # absolute positions now, while only the early owners have arrays.
            for late_cache in cache[early_count:owner_count]:
                late_cache.offset = cache[0].offset
            return cache[:early_count] + cache[owner_count:]

    def prepare_inputs(self, input_ids, pixel_values, image_grid_thw, image_offsets):
        """Encode images once and splice their features into one prompt's embeddings."""
        if self.args.vision_config is None:
            raise ValueError("image inputs require vision_config")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("image inputs currently support one prompt per call")
        if image_grid_thw is None or image_offsets is None:
            raise ValueError("pixels require image_grid_thw and image_offsets")
        grids = (
            image_grid_thw.tolist()
            if hasattr(image_grid_thw, "tolist")
            else image_grid_thw
        )
        positions = build_mrope_positions(input_ids.shape[1], grids, image_offsets)
        features = self.model.visual(pixel_values, image_grid_thw)
        embeddings = self.model.embedding(mx.maximum(input_ids, 0))
        consumed = 0
        for start, end in image_offsets:
            if not bool(
                mx.all(input_ids[0, start : end + 1] == self.args.image_token_id)
            ):
                raise ValueError("image offsets must point to image placeholder tokens")
            count = end - start + 1
            embeddings[:, start : end + 1] = features[
                consumed : consumed + count
            ].astype(embeddings.dtype)
            consumed += count
        return {"input_embeddings": embeddings, "positions": positions}

    @property
    def layers(self):
        return self.model.layers

    def load_weights(self, file_or_weights, strict=True):
        super().load_weights(file_or_weights, strict=strict)
        for layer in self.layers:
            out_proj = layer.self_attn.out_proj
            if isinstance(out_proj, _W8A8QuantizedLinear):
                out_proj._prepare_w8a8()
        return self

    def set_swa_prefill_kv_skip(self, enabled: bool):
        """Select the specialized SWA prefill for eligible X3 attention calls."""
        for layer in self.layers:
            layer.self_attn.swa_prefill_kv_skip = enabled
        self.args.swa_prefill_kv_skip = enabled
        return self

    @property
    def quant_predicate(self):
        # Keep the reference ViT and head gates at checkpoint precision.
        return lambda path, module: (
            not (path.startswith("model.visual.") or path.endswith("self_attn.g_proj"))
        )

    def make_cache(self):
        caches = [
            RotatingKVCache(max_size=self.args.sliding_window, keep=0)
            if self.layers[i].is_sliding
            else KVCache()
            for i in self.model.cache_layer_ids
        ]
        if self.args.uses_mrope:
            caches.append(ArraysCache(1))
        return caches

    def sanitize(self, weights):
        weights = {
            ("model." + name if name.startswith("visual.") else name): value
            for name, value in weights.items()
        }
        ignored_kv = tuple(
            f"model.layers.{i}.self_attn.{projection}."
            for i in range(self.args.num_hidden_layers)
            if self.args.is_ced_consumer(i)
            for projection in ("k_proj", "v_proj")
        )
        weights = {
            name: value
            for name, value in weights.items()
            if not name.startswith(("draft_model.", *ignored_kv))
        }
        if any(".self_attn.q_k_v_proj." in name for name in weights):
            raise ValueError("Spark3 requires factorized Q and separate K/V weights")
        if (
            "model.embed_tokens.weight" in weights
            and "model.embedding.weight" not in weights
        ):
            weights["model.embedding.weight"] = weights.pop("model.embed_tokens.weight")
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return {
            name: value
            for name, value in weights.items()
            if "rotary_emb.inv_freq" not in name
        }
