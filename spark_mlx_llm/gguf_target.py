"""Normalize Spark target GGUF metadata and decode weight rows on CPU.

GGUF quantization is decoded with llama.cpp's official gguf package. Normalized
BF16 row descriptors feed the existing mixed MLX converter; no full decoded
checkpoint or dense model has to be held in memory.
"""

import json
import math
from collections import Counter
from pathlib import Path

# Spark's published spark2_5 BPE pre-tokenizer. GGUF names the profile instead
# of storing its regex pipeline, so reconstruct the same fixed profile.
_PRETOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": "\\p{N}{1,3}"},
            "behavior": "Isolated",
            "invert": False,
        },
        {
            "type": "Split",
            "pattern": {"Regex": "[一-龥\u3040-ゟ゠-ヿ]+"},
            "behavior": "Isolated",
            "invert": False,
        },
        {
            "type": "Split",
            "pattern": {
                "Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n"
                "\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
                "?[\\p{P}\\p{S}]+|[\r\n"
                "]|\\s+(?!\\S)|\\s+"
            },
            "behavior": "Isolated",
            "invert": False,
        },
        {"type": "Digits", "individual_digits": True},
        {
            "type": "ByteLevel",
            "add_prefix_space": False,
            "trim_offsets": True,
            "use_regex": False,
        },
    ],
}


def _tokenizer_assets(meta, config):
    if (
        meta.get("tokenizer.ggml.model") != "gpt2"
        or meta.get("tokenizer.ggml.pre") != "spark2_5"
    ):
        raise ValueError("Target GGUF requires the Spark spark2_5 BPE tokenizer")
    tokens = meta.get("tokenizer.ggml.tokens")
    kinds = meta.get("tokenizer.ggml.token_type")
    merges = meta.get("tokenizer.ggml.merges")
    if (
        not isinstance(tokens, list)
        or len(tokens) != config["vocab_size"]
        or len(set(tokens)) != len(tokens)
    ):
        raise ValueError("Invalid GGUF vocabulary")
    if (
        not isinstance(kinds, list)
        or len(kinds) != len(tokens)
        or not isinstance(merges, list)
    ):
        raise ValueError("Missing GGUF token types or BPE merges")
    pairs = [merge.split(" ", 1) for merge in merges]
    if any(len(pair) != 2 for pair in pairs):
        raise ValueError("Invalid GGUF BPE merge")
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": kinds[i] == 3,
            }
            for i, token in enumerate(tokens)
            if kinds[i] in (3, 4)
        ],
        "normalizer": {"type": "Sequence", "normalizers": []},
        "pre_tokenizer": _PRETOKENIZER,
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": False,
            "use_regex": True,
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": tokens[meta.get("tokenizer.ggml.unknown_token_id", 5)],
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {token: i for i, token in enumerate(tokens)},
            "merges": pairs,
        },
    }
    template = meta.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("Target GGUF needs an embedded chat template")
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        # Keep the published Spark regex pipeline; the optional Mistral repair
        # is inapplicable to this non-Mistral tokenizer.
        "fix_mistral_regex": False,
        "chat_template": template,
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "model_max_length": config["max_position_embeddings"],
    }
    for name in ("bos", "eos", "padding", "unknown"):
        token_id = meta.get(f"tokenizer.ggml.{name}_token_id")
        if token_id is not None:
            if type(token_id) is not int or not 0 <= token_id < len(tokens):
                raise ValueError(f"Invalid GGUF {name} token ID")
            dest = {"padding": "pad", "unknown": "unk"}.get(name, name)
            tokenizer_config[dest + "_token"] = tokens[token_id]
    assets = {
        "tokenizer.json": tokenizer,
        "tokenizer_config.json": tokenizer_config,
        "generation_config.json": {
            name: config[name]
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
            if name in config
        },
    }
    return {
        name: (json.dumps(value, ensure_ascii=False) + "\n").encode()
        for name, value in assets.items()
    }


def read_target_gguf(path):
    """Return normalized BF16 config/descriptors/assets without decoding weights."""
    from gguf import GGMLQuantizationType, GGUFEndian, GGUFReader

    path = Path(path).resolve()
    reader = GGUFReader(str(path))
    if reader.endianess != GGUFEndian.LITTLE:
        raise ValueError("Spark target import currently requires little-endian GGUF")
    meta = {key: field.contents() for key, field in reader.fields.items()}
    if meta.get("general.architecture") != "spark2_5":
        raise ValueError("Expected a Spark spark2_5 target GGUF")

    def field(name):
        key = "spark2_5." + name
        if key not in meta:
            raise ValueError(f"Missing GGUF metadata: {key}")
        return meta[key]

    head_dim = field("attention.key_length")
    if head_dim != field("attention.value_length"):
        raise ValueError("Spark target requires equal key/value head dimensions")
    pattern = field("attention.sliding_window_pattern")
    layers = field("block_count")
    if (
        not isinstance(pattern, list)
        or len(pattern) != layers
        or any(type(value) is not bool for value in pattern)
    ):
        raise ValueError("Invalid GGUF sliding-attention pattern")
    names = {tensor.name for tensor in reader.tensors}
    config = {
        "model_type": "spark2_5",
        "architectures": ["Spark2_5ForCausalLM"],
        "hidden_size": field("embedding_length"),
        "intermediate_size": field("feed_forward_length"),
        "num_hidden_layers": layers,
        "num_attention_heads": field("attention.head_count"),
        "num_key_value_heads": field("attention.head_count_kv"),
        "head_dim": head_dim,
        "vocab_size": field("vocab_size"),
        "max_position_embeddings": field("context_length"),
        "sliding_window": field("attention.sliding_window"),
        "layer_types": [
            "sliding_attention" if value else "full_attention" for value in pattern
        ],
        "rms_norm_eps": field("attention.layer_norm_rms_epsilon"),
        "rope_parameters": {
            "full_attention": {
                "rope_theta": field("rope.freq_base"),
                "partial_rotary_factor": field("rope.dimension_count") / head_dim,
            },
            "sliding_attention": {
                "rope_theta": field("rope.freq_base_swa"),
                "partial_rotary_factor": field("rope.dimension_count_swa") / head_dim,
            },
        },
        "tie_word_embeddings": "output.weight" not in names,
        "attention_bias": False,
        "mlp_bias": False,
        "hidden_act": "gelu",
        "gate_attn_act_mode": "sigmoid",
        "headwise_attn_output_gate": True,
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
    }
    for name, dest in (("bos", "bos"), ("eos", "eos"), ("padding", "pad")):
        key = f"tokenizer.ggml.{name}_token_id"
        if key in meta:
            config[dest + "_token_id"] = meta[key]
    supported = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q4_0,
        GGMLQuantizationType.Q4_1,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q8_0,
    }
    aliases = {
        "token_embd.weight": "model.embedding.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    suffixes = {
        "attn_qkv": "self_attn.q_k_v_proj",
        "attn_output": "self_attn.out_proj",
        "attn_gate": "self_attn.g_proj",
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "ffn_gate": "mlp.gate_proj",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
    }
    for i in range(layers):
        aliases.update(
            {
                f"blk.{i}.{name}.weight": f"model.layers.{i}.{dest}.weight"
                for name, dest in suffixes.items()
            }
        )
    tensors = {}
    for tensor in reader.tensors:
        if tensor.name not in aliases:
            raise ValueError(f"Unsupported Spark target GGUF tensor: {tensor.name}")
        if tensor.tensor_type not in supported:
            raise ValueError(
                f"Unsupported target GGUF tensor type: {tensor.tensor_type.name}"
            )
        shape = list(map(int, reversed(tensor.shape)))
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError(f"Invalid GGUF tensor shape: {tensor.name}")
        tensors[aliases[tensor.name]] = {
            "path": path,
            "dtype": "BF16",
            "shape": shape,
            "bytes": math.prod(shape) * 2,
            "gguf_type": tensor.tensor_type,
            "gguf_data": tensor.data,
        }
    config["gguf_import"] = {
        "source_tensor_types": dict(
            Counter(tensor.tensor_type.name for tensor in reader.tensors)
        ),
        "decode_dtype": "bfloat16",
    }
    return config, tensors, _tokenizer_assets(meta, config)


def read_gguf_rows(desc, first, stop):
    from gguf.quants import dequantize

    from .conversion import bf16

    rows = desc["gguf_data"].reshape(desc["shape"][0], -1)[first:stop]
    values = dequantize(rows, desc["gguf_type"])
    return bf16(values)[0].tobytes()
