"""Regression checks for dev's public APIs with the X3 architecture split."""

import json
import subprocess
import sys
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_model import make_args as spark2_args
from test_spark3 import make_args as spark3_args
from test_spark3_multimodal import TinyTokenizer, tiny_args

from spark_mlx_llm import GenerationConfig, generate, load
from spark_mlx_llm import _target_loader as target_loader
from spark_mlx_llm.cli import build_parser
from spark_mlx_llm.dspark.target import SparkTarget
from spark_mlx_llm.processing_spark3 import Spark3Processor
from spark_mlx_llm.spark2_5 import Model as Spark2Model
from spark_mlx_llm.spark3 import Model as Spark3Model


class Tokenizer(TinyTokenizer):
    eos_token_ids = set()
    has_chat_template = False

    def __init__(self):
        self.detokenizer = self
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = self.decode([token])

    def finalize(self):
        self.last_segment = ""

    def decode(self, tokens):
        return "".join(f"{token} " for token in tokens)


def save_and_load(model, path, monkeypatch, *, nested=False):
    model.save_weights(str(path / "model.safetensors"))
    config = asdict(model.args)
    if nested:
        vision = config.pop("vision_config")
        config = {
            "model_type": "spark3", "text_config": config, "vision_config": vision,
        }
    (path / "config.json").write_text(json.dumps(config))
    tokenizer = Tokenizer()
    monkeypatch.setattr(target_loader, "_load_tokenizer", lambda *args: tokenizer)
    loaded, returned_tokenizer, config = load(path, return_config=True)
    assert returned_tokenizer is tokenizer
    assert config["vocab_size"] == model.args.vocab_size
    return loaded, tokenizer


@pytest.mark.parametrize("architecture", ["spark2_5", "spark3", "spark3_vision"])
def test_public_load_and_generation_for_both_architectures(
    architecture, tmp_path, monkeypatch,
):
    mx.random.seed(21)
    model = (
        Spark2Model(spark2_args()) if architecture == "spark2_5"
        else Spark3Model(tiny_args() if architecture == "spark3_vision" else spark3_args())
    )
    loaded, tokenizer = save_and_load(
        model, tmp_path, monkeypatch, nested=architecture == "spark3_vision"
    )
    before = set(dict(tree_flatten(loaded.parameters())))
    config = GenerationConfig(max_tokens=3, temperature=0, prefill_step_size=2)
    result = generate(loaded, tokenizer, [1, 2, 3], config=config)
    chunks = list(generate(loaded, tokenizer, [1, 2, 3], config=config, stream=True))
    tokens = [1, 2, 3]
    expected = []
    for _ in range(3):
        token = int(mx.argmax(model(mx.array([tokens]))[0, -1]).item())
        expected.append(token)
        tokens.append(token)
    assert tuple(result.token_ids) == tuple(expected)
    assert [c.token_id for c in chunks if c.token_id is not None] == expected
    assert chunks[-1].result.token_ids == result.token_ids
    assert before == set(dict(tree_flatten(loaded.parameters())))


def test_public_spark3_generation_defaults_to_cropped_prefill(tmp_path, monkeypatch):
    model, tokenizer = save_and_load(Spark3Model(spark3_args()), tmp_path, monkeypatch)
    tail_layer = model.layers[4]
    block_type = type(tail_layer)
    forward = block_type.__call__
    rows = []

    def record(layer, inputs, *args, **kwargs):
        if layer is tail_layer:
            rows.append(inputs.shape[1])
        return forward(layer, inputs, *args, **kwargs)

    monkeypatch.setattr(block_type, "__call__", record)
    prompt = [1] * 14
    config = GenerationConfig(max_tokens=1, temperature=0)
    result = generate(model, tokenizer, prompt, config=config)
    assert result.effective_config["prefill_step_size"] == 2048
    assert result.effective_config["ced_prefill_keep_tokens"] == 4
    assert rows == [4, 1]

    rows.clear()
    result = generate(model, tokenizer, prompt, config=config.overlay({"prefill_step_size": 6}))
    assert result.effective_config["prefill_step_size"] == 6
    assert rows == [4, 1, 1]

    rows.clear()
    model.args.ced_prefill_keep_tokens = None
    result = generate(model, tokenizer, prompt, config=config)
    assert result.effective_config["prefill_step_size"] == 512
    assert rows == [13, 1]


def test_image_generation_uses_unified_engine_across_prefill_chunks(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    model, tokenizer = save_and_load(
        Spark3Model(tiny_args()), tmp_path, monkeypatch, nested=True
    )
    processor = Spark3Processor(tokenizer, model.args, {"do_resize": False})
    image = Image.new("RGB", (8, 8), "red")
    calls = []
    original = model.model.visual.__class__.__call__

    def encode(vision, *args, **kwargs):
        calls.append(1)
        return original(vision, *args, **kwargs)

    monkeypatch.setattr(model.model.visual.__class__, "__call__", encode)
    config = GenerationConfig(max_tokens=3, temperature=0, prefill_step_size=2)
    result = generate(model, tokenizer, "a<image>b", images=[image],
                      processor=processor, config=config)
    assert len(calls) == 1
    chunks = list(generate(model, tokenizer, "a<image>b", images=[image],
                           processor=processor, config=config, stream=True))
    assert len(calls) == 2
    assert tuple(c.token_id for c in chunks if c.token_id is not None) == result.token_ids
    data = processor("a<image>b", [image])
    tokens = data.pop("input_ids")
    prepared = model.prepare_inputs(tokens, **data)
    expected = []
    cache = model.make_cache()
    logits = model(tokens, cache=cache, **prepared)
    for _ in range(3):
        token = int(mx.argmax(logits[0, -1]).item())
        expected.append(token)
        logits = model(mx.array([[token]]), cache=cache)
    assert tuple(expected) == result.token_ids


def test_separate_tokenizer_reaches_target_loader(tmp_path, monkeypatch):
    model = Spark3Model(spark3_args())
    save_and_load(model, tmp_path, monkeypatch)
    directory = tmp_path / "tokenizer"
    directory.mkdir()
    seen = []
    tokenizer = Tokenizer()

    def load_tokenizer(path, config, eos):
        seen.append(path)
        return tokenizer

    monkeypatch.setattr(target_loader, "_load_tokenizer", load_tokenizer)
    _, returned = load(tmp_path, tokenizer_path=directory)
    assert returned is tokenizer
    assert seen == [directory]


def test_dev_target_adapter_uses_renamed_spark2_model():
    from spark_mlx_llm.model import Model

    assert Model is Spark2Model
    model = Model(spark2_args())
    target = SparkTarget(model, [0, 1])
    state = target.make_state()
    tokens = mx.array([[1, 2, 3]])
    output = target.run(tokens, state)
    np.testing.assert_allclose(np.array(output.logits), np.array(model(tokens)), atol=1e-6)
    assert len(output.taps) == 2
    state.finish()
    assert state.offset == 3


def test_merged_dspark_checkpoint_retains_public_loading_and_decoding(tmp_path, monkeypatch):
    from spark_mlx_llm.dspark import loader as draft_loader
    from spark_mlx_llm.dspark.drafter import DSparkDrafter

    model = Spark2Model(spark2_args())
    raw = {
        **asdict(model.args), "num_draft_layers": 1, "dspark_block_size": 2,
        "mask_token_id": 0, "target_layer_ids": [0, 1], "markov_rank": 4,
    }
    drafter = DSparkDrafter.from_spark_config(raw)
    weights = dict(tree_flatten(model.parameters()))
    weights.update({
        "draft_model." + name: value for name, value in tree_flatten(drafter.parameters())
    })
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    tokenizer = Tokenizer()
    monkeypatch.setattr(draft_loader, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    loaded, tokenizer = load(tmp_path, draft_quantization="checkpoint")
    assert loaded._spark_runtime.drafter is not None
    common = {"max_tokens": 5, "temperature": 0, "prefill_step_size": 2}
    baseline = generate(loaded, tokenizer, [1, 2, 3],
                        config=GenerationConfig(**common, max_draft_tokens=0))
    speculative = generate(loaded, tokenizer, [1, 2, 3],
                           config=GenerationConfig(**common, max_draft_tokens=2))
    assert speculative.token_ids == baseline.token_ids


def test_spark3_draft_loading_is_rejected_explicitly(tmp_path, monkeypatch):
    save_and_load(Spark3Model(spark3_args()), tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Spark3 speculative decoding"):
        load(tmp_path, draft_model="unused-draft")


def test_cli_keeps_dev_commands_and_x3_image_arguments():
    parser = build_parser()
    args = parser.parse_args([
        "generate", "--model", "x3", "--image", "one.png", "--image", "two.png",
        "--tokenizer-path", "tokenizer", "--no-apply-chat-template", "--stream",
    ])
    assert args.image == ["one.png", "two.png"]
    assert args.tokenizer_path == "tokenizer"
    assert args.apply_chat_template is False
    assert args.stream is True
    assert parser.parse_args(["chat", "--model", "text"]).command == "chat"
    assert parser.parse_args(["server"]).command == "server"


def test_package_and_cli_help_keep_mlx_optional():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import spark_mlx_llm; "
         "from spark_mlx_llm.cli import build_parser; build_parser(); "
         "assert 'mlx.core' not in sys.modules"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
