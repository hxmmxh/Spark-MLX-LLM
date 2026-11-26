"""Self-contained CPU regressions for HTTP startup prefix prefill."""

from contextlib import contextmanager
from dataclasses import replace
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from threading import Thread
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import LRUPromptCache
import pytest

from spark_mlx_llm import GenerationConfig, cli
from spark_mlx_llm import server
from spark_mlx_llm._server_backend import ModelProvider, ResponseGenerator
from spark_mlx_llm._server_prefill import PrefillStage, read_prefix, read_prefixes, spark_cache_key


@pytest.fixture(autouse=True)
def prefill_cpu_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.synchronize()
        mx.set_default_device(previous)


@pytest.fixture
def prefill_base_model():
    """Small real target/draft models plus a consuming streaming detokenizer."""
    from dataclasses import replace

    class Detokenizer:
        def reset(self):
            self.pending = ""

        def add_token(self, token):
            self.pending += f"{token:02d}"

        @property
        def last_segment(self):
            value, self.pending = self.pending, ""
            return value

        def finalize(self):
            pass

    class Tokenizer:
        has_chat_template = True
        eos_token_ids = set()

        def __init__(self):
            self.detokenizer = Detokenizer()
            self.detokenizer.reset()

        def apply_chat_template(self, messages, **kwargs):
            import json
            return "|".join(m["role"] + ":" + m["content"] for m in messages) + json.dumps(kwargs.get("tools", []), sort_keys=True)

        def encode(self, text, **kwargs):
            return [1 + ord(c) % 31 for c in text]

        def decode(self, tokens):
            return "".join(f"{token:02d}" for token in tokens)

    def create(draft, *, head_dim=8):
        import mlx.core as mx
        from spark_mlx_llm.model import Model, ModelArgs
        from spark_mlx_llm.dspark.drafter import DSparkDrafter
        from spark_mlx_llm.dspark.loader import ModelPair
        from spark_mlx_llm.dspark.target import SparkTarget

        mx.random.seed(23)
        raw = {
            "model_type": "spark2_5", "hidden_size": 12, "intermediate_size": 24,
            "vocab_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": head_dim, "num_hidden_layers": 4, "num_draft_layers": 2,
            "dspark_block_size": 4, "mask_token_id": 2, "target_layer_ids": [0, 2],
            "markov_rank": 4, "fc_norm": True, "sliding_window": 4,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000, "partial_rotary_factor": 1},
                "full_attention": {"rope_theta": 5000000, "partial_rotary_factor": 0.25},
            },
            "eos_token_id": None,
        }
        target = SparkTarget(Model(ModelArgs.from_dict(raw)), raw["target_layer_ids"])
        drafter = DSparkDrafter.from_spark_config(raw)
        drafter.bind_lm_head(target.project)
        pair = ModelPair(target, drafter, None, raw, {})
        tokenizer = Tokenizer()
        pair = replace(pair, tokenizer=tokenizer, drafter=pair.drafter if draft else None)
        model = pair.target.model
        import mlx.core as mx
        mx.eval(model.parameters())
        if pair.drafter is not None:
            mx.eval(pair.drafter.parameters())
        object.__setattr__(model, "_spark_runtime", pair)
        object.__setattr__(model, "_spark_load_options", {"model": "tiny", "draft_loaded": draft})
        return model, tokenizer

    return create


@pytest.fixture
def prefill_model(prefill_base_model, monkeypatch):
    """A real TokenizerWrapper so the unmodified upstream HTTP engine can run."""
    from dataclasses import replace
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    import mlx.core as mx
    # The upstream wired-memory guard assumes a Metal device whenever Metal is
    # installed; CPU conformance must not query GPU working-set limits.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)

    class Tokenizer:
        eos_token_id = 31
        bos_token = None
        chat_template = "test"

        def get_vocab(self):
            return {f"{token:02d}": token for token in range(32)}

        def convert_ids_to_tokens(self, token):
            if isinstance(token, list):
                return [self.convert_ids_to_tokens(item) for item in token]
            return f"{token:02d}"

        def encode(self, text, **kwargs):
            if text and len(text) % 2 == 0 and text.isdigit():
                return [int(text[i:i + 2]) for i in range(0, len(text), 2)]
            return [1 + ord(char) % 30 for char in text]

        def decode(self, tokens, **kwargs):
            return "".join(f"{token:02d}" for token in tokens)

        def apply_chat_template(self, messages, tokenize=True, **kwargs):
            text = "|".join(message["role"] + ":" + message["content"] for message in messages)
            return self.encode(text) if tokenize else text

    class Detokenizer:
        def __init__(self, tokenizer):
            self.reset()

        def reset(self):
            self.pending = ""

        def add_token(self, token):
            self.pending += f"{token:02d}"

        @property
        def last_segment(self):
            value, self.pending = self.pending, ""
            return value

        def finalize(self):
            pass

    def create(draft, *, thinking=False, tools=False):
        model, _ = prefill_base_model(draft)
        base = Tokenizer()
        if thinking:
            vocab = {**base.get_vocab(), "<think>": 1, "</think>": 3}
            base.get_vocab = lambda: vocab
        tokenizer = TokenizerWrapper(
            base, detokenizer_class=Detokenizer,
            tool_call_start="04" if tools else None,
            tool_call_end="06" if tools else None,
            tool_parser=(lambda text, tools: {"name": "demo", "arguments": {"value": text}}) if tools else None,
        )
        runtime = replace(model._spark_runtime, tokenizer=tokenizer, generation_config={"eos_token_id": 31})
        object.__setattr__(model, "_spark_runtime", runtime)
        return model, tokenizer

    return create


@pytest.fixture
def conversation_model(prefill_model):
    """Token-round-trippable assistant text and explicit turn delimiters."""
    model, tokenizer = prefill_model(True)
    base = tokenizer._tokenizer

    def encode(text, **kwargs):
        tokens = []
        for piece in re.findall(r"<eot>|[0-9]{2}|.", text, re.DOTALL):
            if piece == "<eot>":
                tokens.append(31)
            elif piece.isdigit() and len(piece) == 2 and int(piece) < 32:
                tokens.append(int(piece))
            else:
                tokens.extend(1 + ord(char) % 30 for char in piece)
        return tokens

    def template(messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = "|".join(message["role"] + ":" + message["content"]
                        + ("<eot>" if message["role"] == "assistant" else "")
                        for message in messages)
        if add_generation_prompt:
            text += "|assistant:"
        return encode(text) if tokenize else text

    base.encode = encode
    base.apply_chat_template = template
    return model, tokenizer


@contextmanager
def running_server(model, tokenizer, config=None, **options):
    config = config or replace(GenerationConfig.http_defaults(), skip_prefill=True)
    handler = server.make_handler(model, tokenizer, config=config, **options)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = Thread(target=lambda: httpd.serve_forever(poll_interval=0.01))
    worker.start()
    connection = HTTPConnection(*httpd.server_address, timeout=10)
    try:
        yield connection
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        handler.close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def post(connection, route, body):
    connection.request("POST", route, json.dumps(body), headers={"Origin": "https://client.example"})
    response = connection.getresponse()
    data = response.read().decode()
    assert response.status == 200, data
    assert response.getheader("Access-Control-Allow-Origin") == "*"
    if body.get("stream"):
        assert data.endswith("data: [DONE]\n\n")
        return [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: {")]
    return [json.loads(data)]


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("seed", [None, 23])
def test_startup_prefill_reuses_kv_and_preserves_output(prefill_model, tmp_path, draft, stream, seed):
    model, tokenizer = prefill_model(draft)
    prefix = "01020304050607080910"
    path = tmp_path / "prefix.txt"
    path.write_text(prefix)
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path), prefill_step_size=3)
    body = {"prompt": prefix + "1112", "max_tokens": 5, "temperature": 0.,
            "stream": stream, "logprobs": True, "top_logprobs": 2}
    if seed is not None:
        body["seed"] = seed
    if stream:
        body["stream_options"] = {"include_usage": True}
    cold_config = replace(config, prefill_file=None, skip_prefill=True)
    with running_server(model, tokenizer, config=cold_config) as connection:
        cold = post(connection, "/v1/completions", body)
    with running_server(model, tokenizer, config=config) as connection:
        for prompt in (prefix + "1112", prefix + "1112", prefix, "131415"):
            warm = post(connection, "/v1/completions", {**body, "prompt": prompt})
            cached = warm[-1]["usage"]["prompt_tokens_details"]["cached_tokens"]
            assert cached >= 9 if prompt.startswith(prefix) else cached == 0
            if prompt == body["prompt"]:
                def approximate(value):
                    if isinstance(value, float):
                        # Cached rotating attention and a full prompt may accumulate
                        # floating-point values in different orders.
                        return pytest.approx(value, abs=5e-4)
                    if isinstance(value, list):
                        return [approximate(item) for item in value]
                    if isinstance(value, dict):
                        return {key: approximate(item) for key, item in value.items()}
                    return value
                assert [packet["choices"] for packet in warm] == approximate([packet["choices"] for packet in cold])


@pytest.mark.parametrize("draft", [False, True])
def test_chat_prefill_uses_rendered_tokens_without_a_second_template(prefill_model, tmp_path, draft):
    model, tokenizer = prefill_model(draft)
    path = tmp_path / "prefix.txt"
    path.write_text("system:fixed")
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path))
    body = {"messages": [{"role": "system", "content": "fixed"}, {"role": "user", "content": "hello"}],
            "max_tokens": 4, "seed": 23}
    with running_server(model, tokenizer, config=config) as connection:
        result = post(connection, "/v1/chat/completions", body)[0]
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == len("system:fixed") - 1


def test_startup_prefill_respects_a_disabled_draft(prefill_model, tmp_path):
    model, tokenizer = prefill_model(True)
    path = tmp_path / "prefix.txt"
    path.write_text("010203040506")
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path), max_draft_tokens=0)
    with running_server(model, tokenizer, config=config) as connection:
        result = post(connection, "/v1/completions", {"prompt": "01020304050607", "max_tokens": 3})[0]
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 5


@pytest.mark.parametrize("setting", ["prefill_file", "skip_prefill", "prefill_newlines", "single_prefill"])
def test_startup_options_cannot_be_changed_by_http(prefill_model, setting):
    model, tokenizer = prefill_model(False)
    with running_server(model, tokenizer) as connection:
        connection.request("POST", "/v1/completions", json.dumps({"prompt": "hello", setting: "anything"}))
        response = connection.getresponse()
        assert response.status == 400
        assert "server startup" in response.read().decode()


@pytest.mark.parametrize("stream", [False, True])
def test_dspark_learns_real_window_prefix_after_startup_mismatch(prefill_model, tmp_path, stream, caplog):
    """Changed window tails must reuse a learned checkpoint, including sliding KV."""
    model, tokenizer = prefill_model(True)
    path = tmp_path / "old-template.txt"
    path.write_text("20212223")
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path), prefill_step_size=3)
    prefix = "01020304050607080910"
    body = {"max_tokens": 5, "temperature": 0, "seed": 23, "stream": stream,
            "logprobs": True, "top_logprobs": 2}
    if stream:
        body["stream_options"] = {"include_usage": True}
    caplog.set_level("INFO")
    with running_server(model, tokenizer, config=config) as connection:
        results = []
        for tail in ("1112", "1314", "1516", "1516"):
            connection.close()  # A different TCP connection, with no conversation/session ID.
            results.append(post(connection, "/v1/completions", {**body, "prompt": prefix + tail}))
    counts = [r[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] for r in results]
    assert counts[:2] == [0, 0]
    assert counts[2:] == [10, 11]
    cold_config = replace(config, prefill_file=None, skip_prefill=True)
    with running_server(model, tokenizer, config=cold_config) as connection:
        cold = post(connection, "/v1/completions", {**body, "prompt": prefix + "1516"})
    def visible(packets):
        return "".join(choice["text"] for packet in packets for choice in packet["choices"])
    assert visible(results[2]) == visible(results[3]) == visible(cold)
    if not stream:
        for warm in results[2:]:
            actual = warm[0]["choices"][0]["logprobs"]["content"]
            expected = cold[0]["choices"][0]["logprobs"]["content"]
            assert [p["id"] for p in actual] == [p["id"] for p in expected]
            assert [p["logprob"] for p in actual] == pytest.approx([p["logprob"] for p in expected], abs=5e-4)
    assert any("cached=10" in record.message and "startup_common=0/3" in record.message for record in caplog.records)


def test_dspark_reuses_system_prefix_for_different_new_window_questions(prefill_model):
    model, tokenizer = prefill_model(True)
    body = {"messages": [{"role": "system", "content": "fixed instructions"},
                         {"role": "user", "content": "hello"}], "max_tokens": 4, "seed": 23}
    with running_server(model, tokenizer) as connection:
        first = post(connection, "/v1/chat/completions", body)[0]
        connection.close()
        changed = {**body, "messages": [body["messages"][0], {"role": "user", "content": "different question"}]}
        second = post(connection, "/v1/chat/completions", changed)[0]
    assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] >= len("system:fixed instructions")


@pytest.mark.parametrize("options", [{"prompt_cache_size": 0}, {"prompt_cache_bytes": 1}])
def test_dspark_runtime_cache_respects_capacity(prefill_model, options):
    model, tokenizer = prefill_model(True)
    body = {"prompt": "010203040506", "max_tokens": 3, "seed": 23}
    with running_server(model, tokenizer, **options) as connection:
        for _ in range(2):
            result = post(connection, "/v1/completions", body)[0]
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_byte_only_cache_retains_more_than_eight_distinct_requests(prefill_model, draft, stream):
    model, tokenizer = prefill_model(draft)
    body = {"max_tokens": 1, "temperature": 0, "seed": 23, "stream": stream,
            "logit_bias": {"31": -100}}
    if stream:
        body["stream_options"] = {"include_usage": True}
    with running_server(model, tokenizer, prompt_cache_size=-1,
                        prompt_cache_bytes=4 * 1024**2) as connection:
        continuations = {}
        for token in range(1, 17):
            prompt = f"{token:02d}0910"
            result = post(connection, "/v1/completions", {**body, "prompt": prompt})
            answer = "".join(choice["text"] for packet in result for choice in packet["choices"])
            # Ordinary upstream sliding KV may already include the answer;
            # test continuation reuse without requiring an unsupported rewind.
            continuations[token] = prompt + answer + "11"
        for token in (1, 2, 3):
            result = post(connection, "/v1/completions", {**body, "prompt": continuations[token]})
            assert result[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] >= 3


def test_byte_only_cache_still_evicts_at_byte_budget(prefill_model, caplog):
    model, tokenizer = prefill_model(True)
    body = {"max_tokens": 1, "temperature": 0, "seed": 23, "logit_bias": {"31": -100}}
    caplog.set_level("INFO")
    with running_server(model, tokenizer, prompt_cache_size=-1,
                        prompt_cache_bytes=40000) as connection:
        post(connection, "/v1/completions", {**body, "prompt": "010910"})
        repeated = post(connection, "/v1/completions", {**body, "prompt": "010910"})[0]
        assert repeated["usage"]["prompt_tokens_details"]["cached_tokens"] >= 2
        for token in (2, 3, 4):
            post(connection, "/v1/completions", {**body, "prompt": f"{token:02d}0910"})
        evicted = post(connection, "/v1/completions", {**body, "prompt": "010910"})[0]
        assert evicted["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    decisions = [record.message for record in caplog.records if "dSpark prompt cache:" in record.message]
    assert decisions
    assert all(int(re.search(r"bytes=(\d+)", message)[1]) <= 40000 for message in decisions)


def test_cli_forwards_unlimited_count_and_keeps_byte_limit(monkeypatch, prefill_model):
    from mlx_lm import server as upstream
    observed = []
    def run_http(host, port, generator, **kwargs):
        assert generator.prompt_cache.max_size == float("inf")
        assert generator.prompt_cache.max_bytes == 40960
        observed.append(True)
    monkeypatch.setattr(upstream, "_run_http_server", run_http)
    def main():
        import sys
        assert sys.argv[sys.argv.index("--prompt-cache-size") + 1] == "-1"
        args = server._namespace("tiny", GenerationConfig.http_defaults(), ("*",),
                                 prompt_cache_size=-1, prompt_cache_bytes=40960)
        provider = upstream.ModelProvider(args)
        provider.preloaded = prefill_model(True)
        generator = upstream.ResponseGenerator(provider, LRUPromptCache(-1))
        upstream._run_http_server("127.0.0.1", 0, generator)
    monkeypatch.setattr(upstream, "main", main)
    assert cli.main(["server", "--model", "tiny", "--skip-prefill",
                     "--prompt-cache-size", "-1", "--prompt-cache-bytes", "40960"]) == 0
    assert observed == [True]


def test_real_prefix_can_replace_startup_cache_with_a_one_entry_budget(prefill_model, tmp_path):
    model, tokenizer = prefill_model(True)
    path = tmp_path / "old-template.txt"
    path.write_text("20212223")
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path))
    body = {"prompt": "010203040506", "max_tokens": 3, "seed": 23,
            "logit_bias": {"31": -100}}
    with running_server(model, tokenizer, config=config, prompt_cache_size=1,
                        prompt_cache_bytes=20000) as connection:
        first = post(connection, "/v1/completions", body)[0]
        # With one entry the completed generation replaces the input-only
        # checkpoint; a continuation must still displace the stale startup KV.
        continuation = body["prompt"] + first["choices"][0]["text"] + "07"
        second = post(connection, "/v1/completions", {**body, "prompt": continuation})[0]
    assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == 8


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("temperature", [0., .7])
@pytest.mark.parametrize("first_limit", [1, 2, 8])
def test_generated_kv_is_reused_in_following_chat_turns(conversation_model, stream, temperature, first_limit):
    model, tokenizer = conversation_model
    messages = [{"role": "system", "content": "fixed instructions"},
                {"role": "user", "content": "hello"}]
    body = {"max_tokens": first_limit, "seed": 23, "temperature": temperature,
            "stream": stream, "logprobs": True, "top_logprobs": 2,
            "logit_bias": {"31": -100}}
    if stream:
        body["stream_options"] = {"include_usage": True}

    def content(packets):
        if not stream:
            return packets[0]["choices"][0]["message"]["content"]
        return "".join(c["delta"].get("content", "") for p in packets for c in p["choices"])

    def comparable(packets):
        def approximate(value):
            if isinstance(value, float):
                return pytest.approx(value, abs=5e-4)
            if isinstance(value, list):
                return [approximate(item) for item in value]
            if isinstance(value, dict):
                return {key: approximate(item) for key, item in value.items()}
            return value
        return approximate([p["choices"] for p in packets])

    followups = []
    with running_server(model, tokenizer, prompt_cache_size=20) as connection:
        first = post(connection, "/v1/chat/completions", {**body, "messages": messages})
        expected_cached = first[-1]["usage"]["prompt_tokens"] + first_limit - 1
        for question in ("continue", "one more"):
            messages += [{"role": "assistant", "content": content(first)},
                         {"role": "user", "content": question}]
            connection.close()
            current = {**body, "max_tokens": 5, "messages": list(messages)}
            warm = post(connection, "/v1/chat/completions", current)
            assert warm[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == expected_cached
            followups.append((current, warm))
            first = warm
            expected_cached = warm[-1]["usage"]["prompt_tokens"] + 4
    for request, warm in followups:
        with running_server(model, tokenizer, prompt_cache_size=20) as connection:
            cold = post(connection, "/v1/chat/completions", request)
        assert [p["choices"] for p in warm] == comparable(cold)


def test_generated_prefixes_for_two_different_conversations_coexist(conversation_model):
    model, tokenizer = conversation_model
    initial = [{"role": "user", "content": "hello"}]
    followups = []
    with running_server(model, tokenizer, prompt_cache_size=20) as connection:
        for forced in (3, 7):
            first = post(connection, "/v1/chat/completions", {
                "messages": initial, "max_tokens": 6, "seed": 23,
                "logit_bias": {str(forced): 100},
            })[0]
            history = [*initial, first["choices"][0]["message"], {"role": "user", "content": "continue"}]
            followups.append((history, first["usage"]["prompt_tokens"] + 5))
        for history, expected in followups:
            result = post(connection, "/v1/chat/completions", {
                "messages": history, "max_tokens": 4, "seed": 23,
            })[0]
            assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == expected


@pytest.mark.parametrize("first_options", [
    {"logit_bias": {"31": 100}},
    {"logit_bias": {"3": 100}, "stop": "0303"},
])
def test_stopped_response_cannot_poison_a_later_chat_prompt(conversation_model, first_options):
    model, tokenizer = conversation_model
    initial = [{"role": "user", "content": "hello"}]
    with running_server(model, tokenizer) as connection:
        first = post(connection, "/v1/chat/completions", {
            "messages": initial, "max_tokens": 8, "seed": 23, **first_options,
        })[0]
        assert first["choices"][0]["finish_reason"] == "stop"
        history = [*initial, first["choices"][0]["message"], {"role": "user", "content": "next"}]
        body = {"messages": history, "max_tokens": 4, "seed": 23, "logprobs": True}
        warm = post(connection, "/v1/chat/completions", body)[0]
    with running_server(model, tokenizer) as connection:
        cold = post(connection, "/v1/chat/completions", body)[0]
    assert warm["choices"][0]["message"] == cold["choices"][0]["message"]
    assert warm["choices"][0]["finish_reason"] == cold["choices"][0]["finish_reason"]


@pytest.mark.parametrize("take", [1, 2, 4])
def test_closed_generation_only_saves_a_materialized_emitted_prefix(conversation_model, take):
    from copy import deepcopy
    from spark_mlx_llm.dspark.generation import _generate_events, _collect, TokenEvent
    model, _ = conversation_model
    pair = replace(model._spark_runtime, generation_config={"eos_token_id": []})
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    saved = []
    def save(tokens, state, caches):
        assert state.offset == len(tokens)
        assert all(c.offset == len(tokens) and not c.pending for c in caches)
        saved.append((list(tokens), *deepcopy((state, caches))))
    events = _generate_events(pair, prompt, max_new_tokens=12, seed=23,
                              _generation_cache_callback=save)
    emitted = []
    for _ in range(take):
        event = next(events)
        assert isinstance(event, TokenEvent)
        emitted.append(event.token_id)
    events.close()
    if take == 1:
        assert len(saved) == 1 and len(saved[0][0]) == len(prompt)
    for prefix, state, caches in saved:
        assert prefix == (prompt + emitted)[:len(prefix)]
        following = prefix + [11, 12]
        warm = _collect(pair, following, max_new_tokens=4, seed=23,
                        _prompt_cache=(prefix, state, caches))
        cold = _collect(pair, following, max_new_tokens=4, seed=23)
        assert warm.token_ids == cold.token_ids


@pytest.mark.parametrize("ending", ["eos", "stop", "close_mid_batch"])
def test_generated_checkpoint_respects_accepted_batch_boundary(conversation_model, monkeypatch, ending):
    from copy import deepcopy
    from spark_mlx_llm.dspark import generation as engine
    from spark_mlx_llm.dspark.sampling import DraftPrediction
    model, _ = conversation_model
    pair = model._spark_runtime
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    proposed = 31 if ending == "eos" else 3
    monkeypatch.setattr(engine, "sample_draft", lambda *a, **kw: DraftPrediction(
        mx.full((1, a[4]), proposed, dtype=mx.int32), None,
    ))

    class Policy:
        def logprobs(self, logits, history):
            token = 31 if ending == "eos" and len(history) > 1 else 3
            return mx.where(mx.arange(32) == token, 0., -100.)

        def sampler(self, logits):
            return mx.argmax(logits, axis=-1)

    saved = []
    def save(tokens, state, caches):
        assert state.offset == len(tokens)
        assert all(c.offset == len(tokens) and not c.pending for c in caches)
        saved.append((list(tokens), *deepcopy((state, caches))))
    events = engine._generate_events(
        pair, prompt, max_new_tokens=12, target_policy=Policy(),
        stop_sequences=((3, 3),) if ending == "stop" else (),
        _generation_cache_callback=save,
    )
    if ending == "close_mid_batch":
        assert next(events).token_id == 3
        assert next(events).token_id == 3
        events.close()
        assert not saved  # The committed batch extends beyond the consumed tokens.
    else:
        result = list(events)[-1]
        assert result.finish_reason == "stop" and result.accepted_tokens == 1
        assert saved[0][0] == prompt + ([3, 31] if ending == "eos" else [3, 3])
        prefix, state, caches = saved[0]
        following = prefix + [11, 12]
        warm = engine._collect(pair, following, max_new_tokens=5, seed=23,
                               _prompt_cache=(prefix, state, caches))
        cold = engine._collect(pair, following, max_new_tokens=5, seed=23)
        assert warm.token_ids == cold.token_ids



def test_bundled_template_integrity_and_newlines(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = GenerationConfig.http_defaults()
    text = read_prefix(config)
    directory = Path(server.__file__).with_name("astudio_template")
    raw = (directory / "template.txt").read_bytes()
    info = json.loads((directory / "template-info.json").read_text())
    assert len(raw) == info["bytes"]
    assert hashlib.sha256(raw).hexdigest() == info["sha256"]
    assert text.endswith(info["prefix_boundary"])
    assert "\r" not in text
    assert read_prefix(replace(config, prefill_newlines="preserve")) == raw.decode("utf-8")


def test_custom_prefix_keeps_whitespace_and_special_markers(tmp_path):
    path = tmp_path / "前缀 with spaces.txt"
    path.write_bytes(b" [im_start]system\r\nhello\rworld \n")
    config = replace(GenerationConfig.http_defaults(), prefill_file=str(path))
    assert read_prefix(config) == " [im_start]system\nhello\nworld \n"
    assert read_prefix(replace(config, prefill_newlines="preserve")) == path.read_bytes().decode()
    assert read_prefix(replace(config, prefill_file=None, skip_prefill=True)) is None
    path.write_text(" \r\n")
    with pytest.raises(ValueError, match="nonempty"):
        read_prefix(config)


@pytest.mark.parametrize("options,message", [
    ({"prefill_file": ""}, "nonempty"),
    ({"prefill_newlines": "crlf"}, "prefill_newlines"),
    ({"skip_prefill": 1}, "boolean"),
    ({"single_prefill": 1}, "boolean"),
    ({"prefill_file": "a", "skip_prefill": True}, "cannot be used together"),
])
def test_invalid_startup_config(options, message):
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**options)


def test_default_asset_corruption_is_reported(monkeypatch, tmp_path):
    from spark_mlx_llm import _server_prefill
    directory = tmp_path / "astudio_template"
    directory.mkdir()
    (directory / "template.txt").write_text("changed")
    (directory / "template-info.json").write_text(json.dumps({"bytes": 7, "sha256": "wrong"}))
    monkeypatch.setattr(_server_prefill, "__file__", str(tmp_path / "_server_prefill.py"))
    with pytest.raises(ValueError, match="does not match"):
        read_prefix(GenerationConfig.http_defaults())


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("options,message", [
    ({"max_context_length": 3}, "leave space"),
    ({"prompt_cache_bytes": 1}, "prompt-cache-bytes"),
    ({"prompt_cache_size": 0}, "prompt-cache-size"),
])
def test_startup_error_reaches_caller_and_thread_exits(prefill_model, draft, options, message):
    config = GenerationConfig.http_defaults()
    options = dict(options)
    if "max_context_length" in options:
        config = replace(config, max_context_length=options.pop("max_context_length"))
    provider = ModelProvider(server._namespace("tiny", config, ("*",), **options),
                             config=config, prefill_text="01020304")
    provider.preloaded = prefill_model(draft)
    generator = ResponseGenerator(provider, LRUPromptCache(provider.cli_args.prompt_cache_size))
    try:
        with pytest.raises(ValueError, match=message):
            generator.wait_ready()
    finally:
        generator.stop_and_join()
    assert not generator._generation_thread.is_alive()


def test_cli_consumes_startup_flags_and_waits_before_http(monkeypatch, tmp_path):
    from mlx_lm import server as upstream
    path = tmp_path / "prefix.txt"
    path.write_text("prefix")
    calls = []
    generator = SimpleNamespace(wait_ready=lambda: calls.append("ready"),
                                stop_and_join=lambda: calls.append("closed"))
    monkeypatch.setattr(upstream, "_run_http_server", lambda *a, **kw: calls.append("http"))
    def main():
        import sys
        assert "--prefill-file" not in sys.argv
        provider = upstream.ModelProvider(server._namespace("tiny", GenerationConfig.http_defaults(), ("*",)))
        assert provider.prefill_text == "prefix"
        upstream._run_http_server("127.0.0.1", 0, generator)
    monkeypatch.setattr(upstream, "main", main)
    assert cli.main(["server", "--model", "tiny", "--prefill-file", str(path)]) == 0
    assert calls == ["ready", "http", "closed"]


def test_cli_failed_prefill_never_starts_http(monkeypatch):
    from mlx_lm import server as upstream
    closed = []
    def failed():
        raise ValueError("prefill failed")
    generator = SimpleNamespace(wait_ready=failed, stop_and_join=lambda: closed.append(True))
    monkeypatch.setattr(upstream, "_run_http_server", lambda *a, **kw: pytest.fail("HTTP started"))
    monkeypatch.setattr(upstream, "main", lambda: upstream._run_http_server("127.0.0.1", 0, generator))
    with pytest.raises(SystemExit):
        cli.main(["server", "--model", "tiny"])
    assert closed == [True]


@pytest.fixture
def startup_stages():
    return (
        PrefillStage("title", "202122232425"),
        PrefillStage("system1", "010203040506"),
        PrefillStage("system1_system2", "01020304050607080910"),
        PrefillStage("system1_system2_user1_fixed", "0102030405060708091011121314"),
    )


@pytest.mark.parametrize("draft", [False, True])
def test_startup_stages_only_compute_the_main_suffix_and_keep_independent_snapshots(
    prefill_model, startup_stages, monkeypatch, draft,
):
    model, tokenizer = prefill_model(draft)
    seen = []
    if draft:
        target = model._spark_runtime.target
        original = target.run
        def run(tokens, *args, **kwargs):
            seen.extend(tokens[0].tolist())
            return original(tokens, *args, **kwargs)
        monkeypatch.setattr(target, "run", run)
    else:
        original = type(model).__call__
        def run(self, tokens, *args, **kwargs):
            seen.extend(tokens[0].tolist())
            return original(self, tokens, *args, **kwargs)
        monkeypatch.setattr(type(model), "__call__", run)
    config = replace(GenerationConfig.http_defaults(), prefill_step_size=3)
    provider = ModelProvider(server._namespace("tiny", config, ("*",), prompt_cache_size=-1),
                             config=config, prefill_text=startup_stages[-1].text,
                             prefill_stages=startup_stages)
    provider.preloaded = model, tokenizer
    cache = LRUPromptCache()
    generator = ResponseGenerator(provider, cache)
    try:
        generator.wait_ready()
        assert seen == [20, 21, 22, 23, 24, *range(1, 14)]
        assert len(cache) == 4
        assert cache.stats_by_type()["system"]["n_sequences"] == (3 if draft else 4)
        key = spark_cache_key(provider.model_key) if draft else provider.model_key
        for stage in startup_stages:
            prefix = tokenizer.encode(stage.text)[:-1]
            snapshot, remaining = cache.fetch_nearest_cache(key, prefix)
            assert snapshot and not remaining
            layers = (*snapshot[0].target.layers, *snapshot[0].draft) if draft else snapshot
            assert all(layer.offset == len(prefix) for layer in layers)
    finally:
        generator.stop_and_join()


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_startup_stage_fallback_and_title_match_cold_output(
    prefill_model, startup_stages, monkeypatch, draft, stream,
):
    model, tokenizer = prefill_model(draft)
    monkeypatch.setattr(server, "read_prefixes", lambda config: () if config.skip_prefill else startup_stages)
    cases = [
        (startup_stages[3].text + "1516", 13),
        (startup_stages[2].text + "1718", 9),
        (startup_stages[1].text + "1918", 5),
        (startup_stages[0].text + "2627", 5),
    ]
    options = {"max_tokens": 4, "temperature": 0., "seed": 23, "stream": stream}
    if stream:
        options["stream_options"] = {"include_usage": True}
    config = replace(GenerationConfig.http_defaults(), prefill_step_size=3)
    results = []
    with running_server(model, tokenizer, config=config, prompt_cache_size=30) as connection:
        for prompt, expected in cases:
            connection.close()
            warm = post(connection, "/v1/completions", {**options, "prompt": prompt})
            assert warm[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == expected
            results.append(warm)
    for (prompt, _), warm in zip(cases, results):
        with running_server(model, tokenizer) as connection:
            cold = post(connection, "/v1/completions", {**options, "prompt": prompt})
        assert [packet["choices"] for packet in warm] == [packet["choices"] for packet in cold]


def test_non_nested_stages_fail_before_any_model_forward(prefill_model, monkeypatch):
    model, tokenizer = prefill_model(True)
    monkeypatch.setattr(model._spark_runtime.target, "run", lambda *a, **k: pytest.fail("Model ran"))
    stages = (PrefillStage("system1", "0102030405"), PrefillStage("system1_system2", "0102030607"))
    config = GenerationConfig.http_defaults()
    provider = ModelProvider(server._namespace("tiny", config, ("*",)), config=config,
                             prefill_text=stages[-1].text, prefill_stages=stages)
    provider.preloaded = model, tokenizer
    generator = ResponseGenerator(provider, LRUPromptCache())
    try:
        with pytest.raises(ValueError, match="nested token prefixes"):
            generator.wait_ready()
    finally:
        generator.stop_and_join()


def test_staged_startup_reports_checkpoints_evicted_by_count_limit(
    prefill_model, startup_stages, caplog,
):
    model, tokenizer = prefill_model(True)
    config = GenerationConfig.http_defaults()
    provider = ModelProvider(server._namespace("tiny", config, ("*",), prompt_cache_size=1),
                             config=config, prefill_text=startup_stages[-1].text,
                             prefill_stages=startup_stages)
    provider.preloaded = model, tokenizer
    cache = LRUPromptCache(1)
    generator = ResponseGenerator(provider, cache)
    try:
        generator.wait_ready()
        assert len(cache) == 1
        assert "Startup prefill retained 1/4 checkpoints" in caplog.text
    finally:
        generator.stop_and_join()


def test_default_plan_uses_raw_byte_cuts_and_single_and_custom_modes(monkeypatch, tmp_path):
    config = GenerationConfig.http_defaults()
    monkeypatch.chdir(tmp_path)
    stages = read_prefixes(config)
    assert [stage.name for stage in stages] == ["title", "system1", "system1_system2", "system1_system2_user1_fixed"]
    assert stages[2].text.startswith(stages[1].text)
    assert stages[3].text.startswith(stages[2].text)
    assert stages[0].text.endswith("<environment_context>")
    assert stages[3].text.endswith("</INSTRUCTIONS><environment_context>")
    single = read_prefixes(replace(config, single_prefill=True))
    assert single == (PrefillStage("main", stages[-1].text),)
    path = tmp_path / "custom.txt"
    path.write_bytes(b"custom\r\nprefix")
    assert read_prefixes(replace(config, prefill_file=str(path))) == (PrefillStage("main", "custom\nprefix"),)
    assert not read_prefixes(replace(config, skip_prefill=True))


@pytest.mark.parametrize("corrupt", ["stage_order", "stage_role", "title_hash", "title_boundary"])
def test_staged_asset_corruption_is_rejected_before_loading(monkeypatch, tmp_path, corrupt):
    from spark_mlx_llm import _server_prefill
    import shutil
    directory = tmp_path / "astudio_template"
    shutil.copytree(Path(_server_prefill.__file__).with_name("astudio_template"), directory)
    path = directory / "template-info.json"
    info = json.loads(path.read_text())
    if corrupt == "stage_order":
        info["main_stages"][1]["bytes"] = info["main_stages"][0]["bytes"]
    elif corrupt == "stage_role":
        info["main_stages"][0]["bytes"] += 1
    elif corrupt == "title_hash":
        (directory / "title.txt").write_text("corrupted")
    else:
        info["title"]["prefix_boundary"] = "incorrect boundary"
    path.write_text(json.dumps(info))
    monkeypatch.setattr(_server_prefill, "__file__", str(tmp_path / "_server_prefill.py"))
    with pytest.raises(ValueError, match="boundaries|boundary|does not match"):
        read_prefixes(GenerationConfig.http_defaults())


def test_cli_single_prefill_is_consumed_before_upstream(monkeypatch):
    from mlx_lm import server as upstream
    def main():
        import sys
        assert "--single-prefill" not in sys.argv
        provider = upstream.ModelProvider(server._namespace("tiny", GenerationConfig.http_defaults(), ("*",)))
        assert [stage.name for stage in provider.prefill_stages] == ["main"]
        assert provider.prefill_text.endswith("<environment_context>")
    monkeypatch.setattr(upstream, "main", main)
    assert cli.main(["server", "--model", "tiny", "--single-prefill"]) == 0
