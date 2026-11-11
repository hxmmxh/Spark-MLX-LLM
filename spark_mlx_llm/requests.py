"""One request/config resolver for HTTP and CLI request replay."""

from dataclasses import fields

from .config import GenerationConfig

_ENVELOPE = {"model", "prompt", "messages", "tools", "stream", "stream_options"}
_CONFIG_FIELDS = {field.name for field in fields(GenerationConfig)}


def resolve_request(body, defaults=None):
    if not isinstance(body, dict):
        raise TypeError("Request must be a JSON object")
    unknown = body.keys() - _ENVELOPE - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown request settings: {sorted(unknown)}")
    if ("prompt" in body) == ("messages" in body):
        raise ValueError("Supply exactly one of prompt or messages")
    prompt = body.get("messages", body.get("prompt"))
    if "messages" in body and (not isinstance(prompt, list) or not prompt):
        raise ValueError("messages must be a nonempty list")
    stream = body.get("stream", False)
    if type(stream) is not bool:
        raise ValueError("stream must be a boolean")
    stream_options = body.get("stream_options", {})
    if not isinstance(stream_options, dict) or stream_options.keys() - {"include_usage"}:
        raise ValueError("stream_options only accepts include_usage")
    if "include_usage" in stream_options and type(stream_options["include_usage"]) is not bool:
        raise ValueError("include_usage must be a boolean")
    defaults = defaults or GenerationConfig()
    config = defaults.overlay({key: value for key, value in body.items() if key in _CONFIG_FIELDS})
    return prompt, body.get("tools"), config, stream
