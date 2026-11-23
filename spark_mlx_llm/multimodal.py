"""Image generation using MLX LM's sampler, chunked prefill and detokenizer."""

from mlx import nn


class _PositionedModel(nn.Module):
    """Bind prompt coordinates to one generation, without modifying the model."""

    def __init__(self, model, positions, input_embeddings=None):
        super().__init__()
        self.model = model
        self._positions = positions
        self._input_embeddings = input_embeddings

    def make_cache(self):
        return self.model.make_cache()

    def __call__(self, inputs, cache=None, input_embeddings=None):
        positions = None
        offset = cache[0].offset if cache is not None else 0
        if (
            input_embeddings is None
            and self._input_embeddings is not None
            and offset < self._input_embeddings.shape[1]
        ):
            input_embeddings = self._input_embeddings[
                :, offset : offset + inputs.shape[1]
            ]
        if input_embeddings is not None:
            positions = self._positions[
                ..., offset : offset + input_embeddings.shape[1]
            ]
        return self.model(
            inputs, cache=cache, input_embeddings=input_embeddings, positions=positions
        )


def generate_multimodal(
    model, processor, prompt, images, *, images_kwargs=None, **kwargs
):
    """Generate from a fresh image prompt; accepts mlx_lm.generate options."""
    from mlx_lm import generate

    if kwargs.get("draft_model") is not None:
        raise ValueError("multimodal speculative decoding is not supported")
    if kwargs.get("prompt_cache") is not None:
        raise ValueError(
            "generate_multimodal starts a fresh prompt; use model calls for cache continuation"
        )
    processed = processor(prompt, images, images_kwargs=images_kwargs)
    ids = processed.pop("input_ids")
    prepared = model.prepare_inputs(ids, **processed)
    bound = _PositionedModel(model, prepared["positions"])
    return generate(
        bound,
        processor.tokenizer,
        prompt=ids[0],
        input_embeddings=prepared["input_embeddings"][0],
        **kwargs,
    )
