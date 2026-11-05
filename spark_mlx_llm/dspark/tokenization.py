"""Prompt encoding and EOS configuration shared by generation and presentation."""


def encode_prompt(
    pair,
    prompt,
    *,
    apply_chat_template=True,
    enable_thinking=True,
    reasoning_effort="low",
):
    """Encode text or validate supplied token IDs against the target vocabulary."""
    if type(enable_thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    if reasoning_effort not in ("low", "high"):
        raise ValueError("reasoning_effort must be low or high")
    if isinstance(prompt, str):
        if pair.tokenizer is None:
            raise ValueError("String prompts need a loaded tokenizer")
        if apply_chat_template and getattr(pair.tokenizer, "has_chat_template", False):
            prompt = pair.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                reasoning_effort=reasoning_effort,
            )
        ids = pair.tokenizer.encode(prompt, add_special_tokens=False)
    else:
        ids = list(prompt)
    if not ids or any(
        type(t) is not int or not 0 <= t < pair.config["vocab_size"] for t in ids
    ):
        raise ValueError("Prompt must contain nonempty, valid target token IDs")
    return ids


def eos_token_ids(pair):
    """Resolve EOS IDs, giving generation_config precedence over model config."""
    value = pair.generation_config.get("eos_token_id", pair.config.get("eos_token_id"))
    if value is None:
        return set()
    return set(value if isinstance(value, list) else [value])
