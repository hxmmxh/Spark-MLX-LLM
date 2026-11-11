"""Internal speculative-decoding components; inference uses the package API."""

from importlib import import_module

from .acceptance import (
    AcceptanceResult,
    greedy_accept,
    sampled_accept,
)
from .checkpoint import (
    CheckpointInfo,
    TensorInfo,
    inspect_checkpoint,
    read_safetensors_header,
    validate_draft_shapes,
)
from .config import DraftLayerSpec, DSparkConfig, spark_layer_specs
from .draft_policy import prefix_survival_probabilities, verification_length
from .metrics import DraftLengthStats, GenerationResult

__all__ = [
    "AcceptanceResult",
    "CheckpointInfo",
    "DSparkConfig",
    "DSparkDrafter",
    "DraftLayerSpec",
    "DraftLengthStats",
    "DraftPrediction",
    "GenerationResult",
    "SamplingOptions",
    "TensorInfo",
    "TokenEvent",
    "greedy_accept",
    "inspect_checkpoint",
    "prefix_survival_probabilities",
    "read_safetensors_header",
    "sample_draft",
    "sampled_accept",
    "save_draft",
    "spark_layer_specs",
    "validate_draft_shapes",
    "verification_length",
]

_RUNTIME_EXPORTS = {
    "DSparkDrafter": "drafter",
    "TokenEvent": "generation",
    "SamplingOptions": "sampling",
    "DraftPrediction": "sampling",
    "sample_draft": "sampling",
    "save_draft": "draft_storage",
}


def __getattr__(name):
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{_RUNTIME_EXPORTS[name]}")
    return getattr(module, name)
