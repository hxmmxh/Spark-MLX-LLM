"""Versioned draft storage rules, usable without importing MLX."""

import hashlib
import json


def module_spec(name, policy):
    """Return the deployed profile for a draft projection, or None if floating."""
    selected = name in ("fc", "lm_head", "markov_head.markov_w2") or (
        name.startswith("layers.")
        and name.endswith(
            (
                ".self_attn.q_k_v_proj",
                ".self_attn.out_proj",
                ".mlp.gate_proj",
                ".mlp.up_proj",
                ".mlp.down_proj",
            )
        )
    )
    if policy == "bf16" or not selected:
        return None
    if policy == "int4" and (name.endswith(".self_attn.q_k_v_proj") or ".mlp." in name):
        return {"bits": 4, "scheme": "affine", "group_size": 32}
    return {"bits": 8, "scheme": "symmetric_per_channel"}


def draft_quantization_config(value):
    """Reject unknown storage formats rather than guessing packed tensor layouts."""
    if value is None:
        return {"policy": "bf16", "modules": {}}
    if (
        not isinstance(value, dict)
        or value.get("format") != "spark-mlx-draft"
        or type(value.get("version")) is not int
        or value["version"] != 1
        or value.get("policy") not in ("bf16", "int8", "int4")
        or not isinstance(value.get("modules"), dict)
    ):
        raise ValueError("Unsupported draft_quantization format/version or policy")
    for name, spec in value["modules"].items():
        if (
            not isinstance(name, str)
            or spec is None
            or spec != module_spec(name, value["policy"])
        ):
            raise ValueError(f"Unsupported draft quantization module: {name!r}")
    return value


def vocabulary_hash(vocabulary):
    encoded = json.dumps(vocabulary, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()
