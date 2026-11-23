# Spark X3 text and vision inference

### Spark3 / Spark X3

`spark_mlx_llm/spark3.py` supports `Spark3ForCausalLM` checkpoints with
`model_type: spark3`. The SWA/FULL model matches the Spark3 vLLM plugin.

The Spark3 model, MLP, attention, RoPE, residual blocks and quantization adapters
are defined independently in this module; it does not import or subclass the
Spark2_5 implementation.

- Factorized query projection: `q_down_proj -> RMSNorm -> q_up_proj -> RoPE`.
  A positive `q_lora_rank` matching the checkpoint is required.
- Separate K/V projections, MQA/GQA, partial non-interleaved RoPE, optional
  head-wise sigmoid output gates, and GELU-gated MLPs.
- Per-layer sliding/full attention and tied or independent output embeddings.
- Optional FULL CED: `kv_source_layer_ids` is one zero-based FULL producer
  index. Every later FULL layer reads its KV, computes its own Q, and omits
  its K/V projections. SWA layers retain independent KV.

The current 9B CED configuration uses 40 layers, hidden size 4096, intermediate
size 12288, 32 Q heads, one KV head, head dimension 256, Q rank 1024, a 512-token
window, and an independent LM head over 131072 tokens. The schedule is
`[SWA, SWA, SWA, FULL] * 10`. Producer layer 19 supplies layers 23, 27, 31, 35
and 39. Checkpoint RoPE settings override the configuration defaults.

`model.make_cache()` returns **35 cache objects** for this model, ordered by
the owning layer indices in `model.model.cache_layer_ids`. Consumers allocate
no cache. This permits native MLX LM prompt-cache save/load without duplicate
storage or updates. Both full-sequence inference and cached chunked prefill /
decode preserve causal attention and absolute RoPE positions across SWA wraps.
Legacy exports containing unused consumer K/V weights are accepted; missing
owner weights still fail strict loading.

Load an unquantized BF16 or already MLX-quantized checkpoint through the same
Python `load()` API and CLI. Model selection follows `config.json`:

```sh
spark-mlx generate --model /path/to/spark3 --prompt '你好' --max-tokens 32
```

For explicit construction, use `from spark_mlx_llm.spark3 import Model,
ModelArgs`, or the package aliases `Spark3Model` and `Spark3ModelArgs`.
The `spark-mlx generate`, `spark-mlx chat` and `spark-mlx server` commands support
both architectures for text inference. Native MLX affine weight quantization
keeps head-gate projections at source precision. The unified `spark-mlx convert`
command remains Spark2_5-only; use the Spark3 quantization tool below for X3
checkpoints.

This implementation does not support QSA, DSpark, or approximate CED prefill
truncation. Unsupported layer types and a non-null `ced_prefill_keep_tokens`
are rejected. It does not use the Spark2_5 fused-QKV extension for Spark3's
factorized query path.

### Spark3 images and mRoPE

Spark3 also loads nested `text_config` plus optional `vision_config`, including
the local `models/test/hf_latest` checkpoint. Explicit root text fields override
nested fields. A single-element CED source list such as `[19]` is accepted.
The existing flat 9B CED checkpoint continues to use its original text path,
ordinary RoPE and 35 owner KV caches.

The MLX vision tower follows `SglangSparkPlugin`: biased Linear patch embedding
after temporal averaging, interpolated absolute positions, adjacent-pair 2D
RoPE, pre-norm Transformer blocks, and a 2x2 spatial merger. It accepts the
checkpoint's `model.visual.*` names and the shorter `visual.*` prefix. Vision
weight shapes are strictly checked. Video and audio are not supported.

The text model reads `mrope_section` and `mrope_interleaved` independently for
SWA/FULL layers, including the interleaved layout used by `hf_latest`. Image
tokens receive T/H/W coordinates; later text and decode retain the resulting
position offset. Each mRoPE request has one additional native `ArraysCache`
holding that offset, so save/load retains it and concurrent requests do not
share position state. These caches cannot be trimmed into an image prefix;
the ordinary 9B cache behavior is unchanged. Approximate prefill truncation is
not enabled.

Install Pillow for image loading and resizing (no PyTorch is required):

```sh
python -m pip install -e '.[vision]'
spark-mlx generate --model ../models/test/hf_latest \
  --image /absolute/path/to/image.png --prompt '请描述这张图片。' --max-tokens 128
```

For the locked uv environment, use `uv sync --locked --extra vision` and run
`.venv/bin/spark-mlx generate`. The unified `generate()` Python API also accepts
`images`, an optional matching `processor`, and `images_kwargs`, and returns the
same `GenerationResult` or streaming chunks as text generation. The CLI supports
`--stream`, `--json` and `--stats` for image prompts. Multimodal requests retain
model-precision KV caches and do not support speculative decoding.

Repeat `--image` for multiple images. The CLI inserts one image placeholder per
image before applying the checkpoint's chat template, unless the prompt already
contains image placeholders. Both one-placeholder-per-image and already expanded
contiguous placeholders are accepted. `--image-min-pixels` and
`--image-max-pixels` control the resize bounds. Supply `--tokenizer-path` if the
checkpoint's matching tokenizer is stored separately.

The supplied `hf_latest` tokenizer vocabulary contains `<|System|>`, `<|User|>`
and `<|Bot|>` (and older `<System>`, `<User>`, `<Bot>` tokens). Its bundled chat
template instead uses `<｜User｜>` / `<｜Assistant｜>`, which are not registered
role tokens in that vocabulary. This can produce repeated role text. The
package preserves the checkpoint template; pass the actual training template
with `--chat-template /path/to/template.jinja`, or supply the exact serialized
training prompt with `--no-apply-chat-template`. A vocabulary alone does not
establish the training prompt's newline, end-of-turn or system-message rules.

Preprocessing reads `preprocessor_config.json` when present; otherwise it uses
Qwen-style patch grouping, Pillow bicubic resize, repeated still frames and RGB
mean/std 0.5, following the SGLang plugin's fallback convention. The processor
validates the patch/temporal/merge geometry against the vision config. Use the
checkpoint's training preprocessing settings for quality evaluation.

The Python image API uses MLX LM's generation/sampling machinery and encodes
images once, including when prefill chunks split an image:

```python
from spark_mlx_llm import load, Spark3Processor, generate_multimodal

model_path = "../models/test/hf_latest"
model, tokenizer = load(model_path)
processor = Spark3Processor.from_pretrained(model_path, tokenizer, model.args)
prompt = processor.image_placeholder + "\n请描述这张图片。"
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True,
)
text = generate_multimodal(
    model, processor, prompt, ["/absolute/path/to/image.png"], max_tokens=128,
)
```

`generate_multimodal` starts a fresh prompt. For explicit cache continuation,
call the model with processor-produced `pixel_values`, `image_grid_thw` and
`image_offsets`, or with prepared embeddings and explicit `[3,batch,length]`
positions. Text-only calls work for both ordinary RoPE and mRoPE checkpoints.
Image requests currently accept one prompt per call; multiple images within it
are supported. The unified chat/server commands remain text interfaces;
their HTTP image routing is not implemented by this package.

Run the numerical/cache tests and opt in to real multimodal header checks:

```sh
python -m pytest tests/ -v
SPARK3_MM_MODEL=../models/test/hf_latest \
  python -m pytest tests/test_spark3_multimodal.py -v
```

Run unit tests and opt in to real-checkpoint integration checks:

```sh
python -m pytest tests/ -v
SPARK3_MODEL=/path/to/spark3 python -m pytest tests/test_spark3_integration.py -v
```

The integration checks inspect every shard's completeness and parameter shapes,
strictly load the checkpoint, and compare cached decode with a full forward.
For BF16, they require equal greedy tokens, relative logit L2 error below 2%,
and output-distribution KL divergence below 0.002. The tiny-model unit tests
separately use a strict FP32-to-FP64 NumPy comparison on CPU, precision-aware
Metal checks, chunked prefill, SWA wraps and cache serialization.

### Quantize BF16 Spark3 to W4 with an 8-bit LM head

[`tools/quantize_spark3_w4_head8.py`](../tools/quantize_spark3_w4_head8.py)
converts local BF16 Spark3 checkpoints directly to MLX affine weights without
calibration. Text matrices and input embeddings use 4 bits, while `lm_head`
uses 8 bits; both use group size 64. Vision weights, attention head gates,
normalization weights and biases remain BF16. Activations are not quantized.
For tied checkpoints, the converter exports independent 4-bit input embeddings
and an 8-bit output head and sets `tie_word_embeddings` to false.

Run from the repository root with the installed MLX environment:

```sh
.venv/bin/python tools/quantize_spark3_w4_head8.py \
    --input-root ../models/test \
    --models 9b_ced_20000 hf_latest second_hf \
    --output-root ../models/quantized_mlx
```

These are the defaults; default paths are resolved relative to the repository,
so the script can also be called from another working directory. Outputs are
named `<model>-w4-head8-g64`. Existing output or `.incomplete` directories are
never overwritten; use a new `--output-root` to repeat a conversion.

The converter processes tensors in batches and writes shards of approximately
1 GiB. Before publishing each output directory, it verifies the saved tensor
shapes and dtypes, byte equality of preserved BF16 tensors, strict model loading,
text prefill and cached decode, and a synthetic image forward for vision models.
Each model includes `quantization_report.json`; the output root receives
`quantization_summary.json` after all requested models finish. These checks
verify loading and execution, without evaluating generation quality.
