"""Compatibility imports for the Spark2_5 model, now in spark2_5."""

from .spark2_5 import (
    MLP,
    Attention,
    Model,
    ModelArgs,
    Spark2_5Model,
    TransformerBlock,
)

__all__ = [
    "MLP",
    "Attention",
    "Model",
    "ModelArgs",
    "Spark2_5Model",
    "TransformerBlock",
]
