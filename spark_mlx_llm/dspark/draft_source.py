"""Read standalone Spark drafts without MLX or a GGUF Python dependency.

Tensor descriptors address floating-point or deployed MLX integer parameters. Q/K/V
concatenation and stacked feature norms are represented as views, so conversion
can stream them without loading the full draft into memory.
"""

import copy
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from .checkpoint import (
    CheckpointInfo,
    TensorInfo,
    inspect_checkpoint,
    validate_draft_shapes,
)
from .config import DSparkConfig
from .quantization_format import vocabulary_hash

_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "U32": 4}
_GEOMETRY = (
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)
_DRAFT_FIELDS = (
    "num_draft_layers",
    "dspark_block_size",
    "mask_token_id",
    "target_layer_ids",
    "draft_vocab_size",
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "fc_norm",
    "dspark_tie_word_embeddings",
    "sample_from_anchor",
    "logits_start",
    "dspark_shared_embedding",
    "dspark_readout",
    "draft_quantization",
    "dspark_target_vocab_sha256",
)


def read_tensor(desc, first=0, stop=None):
    """Read rows from a contiguous tensor or a concatenation of row slices."""
    stop = desc["shape"][0] if stop is None else stop
    if not 0 <= first <= stop <= desc["shape"][0]:
        raise ValueError("Invalid source tensor row range")
    if "parts" in desc:
        pieces, offset = [], 0
        for part in desc["parts"]:
            rows = part["shape"][0]
            begin, end = max(first - offset, 0), min(stop - offset, rows)
            if begin < end:
                pieces.append(read_tensor(part, begin, end))
            offset += rows
        return b"".join(pieces)
    stride = desc["bytes"] // desc["shape"][0]
    with desc["path"].open("rb") as stream:
        stream.seek(desc["offset"] + first * stride)
        data = stream.read((stop - first) * stride)
    if len(data) != (stop - first) * stride:
        raise ValueError("Draft source changed or was truncated")
    return data


@dataclass(frozen=True)
class DraftSource:
    config: dict
    tensors: dict
    files: tuple[Path, ...]
    source_format: str

    def checkpoint(self):
        return CheckpointInfo(
            self.files[0].parent,
            tuple(
                TensorInfo(
                    "draft_model." + name,
                    tuple(desc["shape"]),
                    desc["dtype"],
                    desc["path"] if "path" in desc else desc["parts"][0]["path"],
                    (0, desc["bytes"]),
                )
                for name, desc in self.tensors.items()
            ),
        )


class _GGUF:
    """Bounded little-endian GGUF v2/v3 metadata and float tensor reader."""

    _FORMATS: ClassVar[dict[int, str]] = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def __init__(self, path):
        self.path = path
        self.size = path.stat().st_size
        with path.open("rb") as self.stream:
            if self.take(4) != b"GGUF" or self.number("I") not in (2, 3):
                raise ValueError("Expected a little-endian GGUF v2/v3 draft")
            count, metadata_count = self.number("Q"), self.number("Q")
            if not 0 < count <= 100000 or metadata_count > 100000:
                raise ValueError("Invalid GGUF tensor/metadata count")
            self.metadata = {}
            for _ in range(metadata_count):
                key = self.string()
                if key in self.metadata:
                    raise ValueError(f"Duplicate GGUF metadata key: {key}")
                self.metadata[key] = self.value(self.number("I"))
            self.tensors = {}
            for _ in range(count):
                name, ndim = self.string(), self.number("I")
                if name in self.tensors or not 1 <= ndim <= 4:
                    raise ValueError(f"Invalid or duplicate GGUF tensor: {name}")
                shape = list(reversed([self.number("Q") for _ in range(ndim)]))
                kind, offset = self.number("I"), self.number("Q")
                if kind not in (0, 1, 30):
                    raise ValueError(
                        f"Quantized GGUF draft tensor {name!r} (type={kind}) is unsupported; "
                        "use the floating-point BF16 release"
                    )
                if any(dim <= 0 for dim in shape):
                    raise ValueError(f"Invalid GGUF shape: {name}")
                dtype = {0: "F32", 1: "F16", 30: "BF16"}[kind]
                self.tensors[name] = {
                    "path": path,
                    "offset": offset,
                    "shape": shape,
                    "dtype": dtype,
                    "bytes": math.prod(shape) * _SIZES[dtype],
                }
            alignment = self.metadata.get("general.alignment", 32)
            if (
                type(alignment) is not int
                or not 0 < alignment <= 4096
                or alignment & (alignment - 1)
            ):
                raise ValueError("Invalid GGUF alignment")
            base = (self.stream.tell() + alignment - 1) // alignment * alignment
            end = base
            for desc in sorted(self.tensors.values(), key=lambda d: d["offset"]):
                if desc["offset"] % alignment:
                    raise ValueError("Unaligned GGUF tensor")
                desc["offset"] += base
                if desc["offset"] < end or desc["offset"] + desc["bytes"] > self.size:
                    raise ValueError("Overlapping or truncated GGUF tensor data")
                end = desc["offset"] + desc["bytes"]

    def take(self, length):
        # Metadata and tensor headers are bounded independently of payload size.
        if length > 64 * 1024**2 or self.stream.tell() + length > min(
            self.size, 64 * 1024**2
        ):
            raise ValueError("Invalid or excessive GGUF header size")
        data = self.stream.read(length)
        if len(data) != length:
            raise ValueError("Truncated GGUF header")
        return data

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.number("Q")).decode("utf-8")

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            if subtype == 9 or count > 2000000:
                raise ValueError("Unsupported GGUF array")
            return [self.value(subtype) for _ in range(count)]
        if kind not in self._FORMATS:
            raise ValueError(f"Unknown GGUF metadata type: {kind}")
        return self.number(self._FORMATS[kind])


def _vocab(path):
    path = path / "tokenizer.json"
    if not path.is_file():
        return None
    tokenizer = json.loads(path.read_text())
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise TypeError("Spark draft compatibility requires a BPE tokenizer vocabulary")
    vocab = dict(vocab)
    vocab.update(
        {token["content"]: token["id"] for token in tokenizer.get("added_tokens", [])}
    )
    return vocab


def _merge_config(target, draft):
    if draft.get("model_type", "spark2_5") != "spark2_5":
        raise ValueError("Expected a Spark DSpark draft configuration")
    for key in _GEOMETRY:
        if key in draft and draft[key] != target.get(key):
            raise ValueError(
                f"Draft/target {key} mismatch: {draft[key]} != {target.get(key)}"
            )
    for key in ("rms_norm_eps", "sliding_window"):
        expected = target.get(key, 1e-6 if key == "rms_norm_eps" else None)
        if key in draft and not math.isclose(draft[key], expected, rel_tol=1e-6):
            raise ValueError(f"Draft/target {key} mismatch")
    if "rope_parameters" in draft and draft["rope_parameters"].get(
        "sliding_attention"
    ) != target.get("rope_parameters", {}).get("sliding_attention"):
        raise ValueError("Draft/target sliding-attention RoPE mismatch")
    result = copy.deepcopy(target)
    # Do not inherit stale draft fields from a different merged checkpoint.
    for key in _DRAFT_FIELDS:
        result.pop(key, None)
        if key in draft:
            result[key] = copy.deepcopy(draft[key])
    DSparkConfig.from_spark_dict(result)
    if type(result.get("dspark_shared_embedding", False)) is not bool:
        raise ValueError("dspark_shared_embedding must be a boolean")
    if result.get("dspark_readout", "target") not in ("target", "embedding"):
        raise ValueError("dspark_readout must be target or embedding")
    return result


def _from_gguf(path, target, target_directory, target_vocabulary=None):
    source = _GGUF(path)
    meta, tensors = source.metadata, source.tensors
    if (
        meta.get("general.architecture") != "dflash"
        or "markov_w1.weight" not in tensors
    ):
        raise ValueError("Expected a Spark DSpark GGUF with a vanilla Markov head")

    def field(name):
        key = "dflash." + name
        if key not in meta:
            raise ValueError(f"Missing GGUF metadata: {key}")
        return meta[key]

    pattern = field("attention.sliding_window_pattern")
    if (
        field("attention.causal") is not False
        or not isinstance(pattern, list)
        or len(pattern) != field("block_count")
        or any(value is not True for value in pattern)
    ):
        raise ValueError(
            "Spark DSpark requires bidirectional draft blocks and all-SWA layers"
        )
    if field("hidden_activation") != "gelu":
        raise ValueError("Spark DSpark requires GELU")
    if field("attention.key_length") != field("attention.value_length"):
        raise ValueError("Spark DSpark requires equal key/value head dimensions")
    swa = target.get("rope_parameters", {}).get("sliding_attention", {})
    if not math.isclose(
        field("rope.freq_base_swa"), swa.get("rope_theta", 10000), rel_tol=1e-6
    ) or field("rope.dimension_count_swa") != int(
        target["head_dim"] * swa.get("partial_rotary_factor", 1)
    ):
        raise ValueError("Draft/target sliding-attention RoPE mismatch")
    draft = {
        "hidden_size": field("embedding_length"),
        "intermediate_size": field("feed_forward_length"),
        "vocab_size": field("vocab_size"),
        "num_attention_heads": field("attention.head_count"),
        "num_key_value_heads": field("attention.head_count_kv"),
        "head_dim": field("attention.key_length"),
        "num_draft_layers": field("block_count"),
        "dspark_block_size": field("block_size"),
        # GGUF stores the next layer's input; Spark uses the preceding layer's output.
        "target_layer_ids": [i - 1 for i in field("target_layers")],
        "mask_token_id": meta.get("tokenizer.ggml.mask_token_id"),
        "rms_norm_eps": field("attention.layer_norm_rms_epsilon"),
        "sliding_window": field("attention.sliding_window"),
        "sample_from_anchor": field("sample_from_anchor"),
        "fc_norm": "fc_norm.weight" in tensors,
        "markov_rank": tensors["markov_w1.weight"]["shape"][1],
        "enable_confidence_head": "conf_proj.weight" in tensors,
        "dspark_shared_embedding": "token_embd.weight" not in tensors,
        "dspark_readout": "target",
    }
    config = _merge_config(target, draft)
    tokens = meta.get("tokenizer.ggml.tokens")
    if tokens is not None and (
        target_directory is not None or target_vocabulary is not None
    ):
        vocab = (
            target_vocabulary
            if target_vocabulary is not None
            else _vocab(target_directory)
        )
        if vocab is not None and (
            len(tokens) != len(vocab)
            or any(vocab.get(token) != i for i, token in enumerate(tokens))
        ):
            raise ValueError("Draft/target tokenizer token IDs differ")
    aliases = {
        "fc.weight": "fc.weight",
        "enc.output_norm.weight": "hidden_norm.weight",
        "output_norm.weight": "norm.weight",
        "token_embd.weight": "embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "markov_w1.weight": "markov_head.markov_w1.weight",
        "markov_w2.weight": "markov_head.markov_w2.weight",
        "conf_proj.weight": "confidence_head.proj.weight",
        "conf_proj.bias": "confidence_head.proj.bias",
    }
    result, consumed = {}, set()
    for name, dest in aliases.items():
        if name in tensors:
            result[dest] = tensors[name]
            consumed.add(name)
    if "fc_norm.weight" in tensors:
        desc = tensors["fc_norm.weight"]
        expected = [len(config["target_layer_ids"]), config["hidden_size"]]
        if desc["shape"] != expected:
            raise ValueError("Invalid stacked GGUF feature norms")
        stride = config["hidden_size"] * _SIZES[desc["dtype"]]
        for i in range(expected[0]):
            result[f"fc_norm.{i}.weight"] = {
                **desc,
                "shape": expected[1:],
                "offset": desc["offset"] + i * stride,
                "bytes": stride,
            }
        consumed.add("fc_norm.weight")
    layers = {
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attn_gate": "self_attn.g_proj",
        "attn_output": "self_attn.out_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    for i in range(config["num_draft_layers"]):
        for name, dest in layers.items():
            key = f"blk.{i}.{name}.weight"
            if key not in tensors:
                raise ValueError(f"Missing GGUF tensor: {key}")
            result[f"layers.{i}.{dest}.weight"] = tensors[key]
            consumed.add(key)
        keys = [f"blk.{i}.attn_{name}.weight" for name in ("q", "k", "v")]
        if any(key not in tensors for key in keys):
            raise ValueError(f"Missing GGUF Q/K/V tensors for layer {i}")
        parts = [tensors[key] for key in keys]
        widths = [
            config["num_attention_heads"] * config["head_dim"],
            config["num_key_value_heads"] * config["head_dim"],
        ]
        widths.append(widths[-1])
        if (
            any(
                part["shape"] != [width, config["hidden_size"]]
                for part, width in zip(parts, widths)
            )
            or len({part["dtype"] for part in parts}) != 1
        ):
            raise ValueError(f"Invalid GGUF Q/K/V geometry or dtype for layer {i}")
        result[f"layers.{i}.self_attn.q_k_v_proj.weight"] = {
            "parts": parts,
            "dtype": parts[0]["dtype"],
            "shape": [sum(widths), config["hidden_size"]],
            "bytes": sum(part["bytes"] for part in parts),
        }
        consumed.update(keys)
    if consumed != tensors.keys():
        raise ValueError(
            f"Unsupported GGUF draft tensors: {sorted(tensors.keys() - consumed)}"
        )
    return DraftSource(config, result, (path,), "gguf")


def resolve_draft_source(
    path, target_config, *, target_directory=None, target_vocabulary=None
):
    """Normalize a floating/quantized MLX Safetensors draft or floating Spark GGUF."""
    path = Path(path).expanduser().resolve()
    target_directory = Path(target_directory) if target_directory is not None else None
    if path.is_file() and path.suffix.lower() == ".gguf":
        source = _from_gguf(path, target_config, target_directory, target_vocabulary)
    elif path.is_dir():
        raw = json.loads((path / "config.json").read_text())
        if "num_draft_layers" not in raw and "num_hidden_layers" in raw:
            raw["num_draft_layers"] = raw["num_hidden_layers"]
        config = _merge_config(target_config, raw)
        checkpoint = inspect_checkpoint(path)
        tensors = {}
        for tensor in checkpoint.tensors:
            name = tensor.name.removeprefix("draft_model.").removeprefix("model.")
            if name in tensors or tensor.dtype not in _SIZES:
                raise ValueError(
                    f"Duplicate or unsupported draft tensor: {tensor.name}"
                )
            # Safetensors offsets are relative to the payload start.
            with tensor.shard.open("rb") as stream:
                base = 8 + struct.unpack("<Q", stream.read(8))[0]
            tensors[name] = {
                "path": tensor.shard,
                "offset": base + tensor.data_offsets[0],
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "bytes": tensor.nbytes,
            }
        files = {tensor.shard for tensor in checkpoint.tensors} | {path / "config.json"}
        if target_directory is not None or target_vocabulary is not None:
            other = _vocab(path)
            target = (
                target_vocabulary
                if target_vocabulary is not None
                else _vocab(target_directory)
            )
            if other is not None and target is not None and other != target:
                raise ValueError("Draft/target tokenizer token IDs differ")
            fingerprint = config.get("dspark_target_vocab_sha256")
            if (
                fingerprint is not None
                and target is not None
                and fingerprint != vocabulary_hash(target)
            ):
                raise ValueError(
                    "Draft/target tokenizer token IDs differ (vocabulary hash)"
                )
            if (path / "tokenizer.json").is_file():
                files.add(path / "tokenizer.json")
        source = DraftSource(config, tensors, tuple(sorted(files)), "safetensors")
    else:
        raise ValueError(
            f"Draft source must be a Safetensors directory or a BF16 .gguf file: {path}"
        )
    validate_draft_shapes(
        DSparkConfig.from_spark_dict(source.config),
        source.checkpoint(),
        shared_embedding=source.config.get("dspark_shared_embedding", False),
        quantization=source.config.get("draft_quantization"),
    )
    return source
