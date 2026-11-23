"""Spark conversion and inference APIs, loaded on demand to keep MLX optional."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .conversion import convert_to_mlx
    from .generation import (
        GenerationChunk,
        GenerationConfig,
        GenerationResult,
        generate,
    )
    from .loader import load
    from .multimodal import generate_multimodal
    from .processing_spark3 import Spark3Processor
    from .spark2_5 import Model, ModelArgs
    from .spark3 import Model as Spark3Model
    from .spark3 import ModelArgs as Spark3ModelArgs

__all__ = [
    "GenerationChunk",
    "GenerationConfig",
    "GenerationResult",
    "Model",
    "ModelArgs",
    "Spark3Model",
    "Spark3ModelArgs",
    "Spark3Processor",
    "convert_to_mlx",
    "generate",
    "generate_multimodal",
    "load",
]


def __getattr__(name):
    if name == "convert_to_mlx":
        from .conversion import convert_to_mlx

        globals()[name] = convert_to_mlx
    elif name == "GenerationConfig":
        from .generation import GenerationConfig

        globals()[name] = GenerationConfig
    elif name in {"Model", "ModelArgs"}:
        from .spark2_5 import Model, ModelArgs

        globals().update(Model=Model, ModelArgs=ModelArgs)
    elif name in {"Spark3Model", "Spark3ModelArgs"}:
        from .spark3 import Model as Spark3Model
        from .spark3 import ModelArgs as Spark3ModelArgs

        globals().update(Spark3Model=Spark3Model, Spark3ModelArgs=Spark3ModelArgs)
    elif name == "Spark3Processor":
        from .processing_spark3 import Spark3Processor

        globals()[name] = Spark3Processor
    elif name == "generate_multimodal":
        from .multimodal import generate_multimodal

        globals()[name] = generate_multimodal
    elif name in {"generate", "GenerationChunk", "GenerationResult"}:
        from .generation import GenerationChunk, GenerationResult, generate

        globals().update(generate=generate, GenerationChunk=GenerationChunk, GenerationResult=GenerationResult)
    elif name == "load":
        from .loader import load

        globals()["load"] = load
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return globals()[name]


def __dir__():
    return sorted(set(globals()) | set(__all__))
