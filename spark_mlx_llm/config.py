"""The shared generation configuration for Python, CLI, HTTP and benchmarks."""

import argparse
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

_KV_QUANTIZATION_FIELDS = (
    "kv_bits",
    "kv_group_size",
    "quantized_kv_start",
    "kv_quantization_scope",
)
_KV_FIELDS = ("prefill_step_size", *_KV_QUANTIZATION_FIELDS)


def _json_object(value):
    try:
        result = json.loads(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid JSON: {error}") from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return result


class _TemplateArguments(Mapping):
    """Read-only snapshot whose nested values are copied on access."""

    def __init__(self, values):
        self._values = deepcopy(dict(values))

    def __getitem__(self, name):
        return deepcopy(self._values[name])

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        return repr(self._values)


@dataclass(frozen=True, kw_only=True)
class GenerationConfig:
    """Sampling, templates and Spark KV defaults, exported by purpose.

    None for thinking settings preserves the model template. KV caches retain
    model precision by default, matching chat/server's upstream behavior.
    KV quantization is opt-in and independent of checkpoint quantization.
    """

    max_tokens: int | None = 512
    max_context_length: int | None = None
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    enable_thinking: bool | None = True
    reasoning_effort: str | None = "low"
    chat_template_kwargs: Mapping = field(default_factory=dict)
    prefill_step_size: int = 512
    kv_bits: int | None = None
    kv_group_size: int = 32
    quantized_kv_start: int = 0
    kv_quantization_scope: str = "full-attention"
    seed: int | None = 0
    draft_temperature: float = 1.0
    draft_top_p: float = 1.0
    draft_top_k: int = 0
    max_draft_tokens: int = 3
    confidence_threshold: float = 0.0
    asynchronous_draft: bool = True
    async_context: bool = True
    compile_draft_chain: bool = True
    apply_chat_template: bool = True
    cache_limit_mib: int | None = 256
    stop_at_eos: bool = True
    clear_cache: bool = False

    @classmethod
    def http_defaults(cls):
        """Original ordinary HTTP defaults; explicit file/CLI values override."""
        return cls(
            temperature=0.0, top_p=1.0, seed=None, enable_thinking=None,
            reasoning_effort=None, prefill_step_size=2048,
        )

    @staticmethod
    def reject_http_kv_options(values):
        """Reject explicit HTTP KV options, even when they match defaults."""
        unsupported = set(_KV_QUANTIZATION_FIELDS).intersection(values)
        if unsupported:
            raise ValueError(
                "HTTP serving does not accept KV configuration: "
                + ", ".join(sorted(unsupported))
            )

    def validate_http(self):
        """A constructed config may only retain the service's default KV state."""
        defaults = self.http_defaults()
        self.reject_http_kv_options({
            name: getattr(self, name) for name in _KV_QUANTIZATION_FIELDS
            if getattr(self, name) != getattr(defaults, name)
        })

    def __post_init__(self):
        for name, minimum in (
            ("top_k", 0),
            ("draft_top_k", 0),
            ("max_draft_tokens", 0),
            ("prefill_step_size", 1),
            ("quantized_kv_start", 0),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name, minimum in (
            ("max_tokens", 1), ("max_context_length", 1),
            ("seed", 0), ("cache_limit_mib", 0),
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"{name} must be None or an integer >= {minimum}")
        for name in ("temperature", "top_p", "min_p", "draft_temperature", "draft_top_p", "confidence_threshold"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name not in ("temperature", "draft_temperature") and value > 1)
                or (name in ("top_p", "draft_top_p") and value == 0)
            ):
                bounds = ">= 0" if name in ("temperature", "draft_temperature") else "between 0 and 1"
                raise ValueError(f"{name} must be a finite number {bounds}")
        for name in ("asynchronous_draft", "async_context", "compile_draft_chain", "apply_chat_template", "stop_at_eos", "clear_cache"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.chat_template_kwargs, Mapping):
            raise TypeError("chat_template_kwargs must be a JSON object")
        extras = dict(self.chat_template_kwargs)
        reserved = {
            "messages",
            "tools",
            "tokenize",
            "add_generation_prompt",
            "enable_thinking", "reasoning_effort",
        } & extras.keys()
        if reserved:
            raise ValueError(f"Reserved chat template arguments: {sorted(reserved)}")
        object.__setattr__(self, "chat_template_kwargs", _TemplateArguments(extras))
        if self.enable_thinking is not None and type(self.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be true, false or null")
        if self.reasoning_effort not in (None, "low", "high"):
            raise ValueError("reasoning_effort must be 'low', 'high' or null")
        if self.kv_bits is not None and (
            type(self.kv_bits) is not int or self.kv_bits not in (4, 8)
        ):
            raise ValueError("kv_bits must be None, 4 or 8")
        if type(self.kv_group_size) is not int or self.kv_group_size not in (
            32,
            64,
            128,
        ):
            raise ValueError("kv_group_size must be 32, 64 or 128")
        if self.kv_quantization_scope not in ("all", "full-attention"):
            raise ValueError("kv_quantization_scope must be 'all' or 'full-attention'")

    def overlay(self, values):
        """Merge a higher-priority layer using canonical field names."""
        if not isinstance(values, dict):
            raise TypeError("Generation configuration must be a JSON object")
        unknown = values.keys() - {item.name for item in fields(self)}
        if unknown:
            raise ValueError(f"Unknown generation settings: {sorted(unknown)}")
        changes = dict(values)
        extras = changes.pop("chat_template_kwargs", {})
        if not isinstance(extras, Mapping):
            raise TypeError("chat_template_kwargs must be a JSON object")
        extras = dict(extras)
        return replace(
            self,
            chat_template_kwargs={**self.chat_template_kwargs, **extras},
            **changes,
        )

    @classmethod
    def from_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, values):
        """Build a configuration, rejecting unknown fields and invalid values."""
        return cls().overlay(values)

    @classmethod
    def from_options(cls, values):
        """Extract configuration fields from mixed options, such as vars(args)."""
        if not isinstance(values, Mapping):
            raise TypeError("Generation options must be a mapping")
        return cls.from_dict(
            {f.name: values[f.name] for f in fields(cls) if f.name in values}
        )

    def sampler_kwargs(self):
        return {
            "temp": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
        }

    def to_dict(self):
        """A complete JSON-safe snapshot; no caller-owned mutable values escape."""
        return {
            item.name: deepcopy(dict(self.chat_template_kwargs))
            if item.name == "chat_template_kwargs" else getattr(self, item.name)
            for item in fields(self)
        }

    def template_kwargs(self):
        result = dict(self.chat_template_kwargs)
        for name in ("enable_thinking", "reasoning_effort"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def add_arguments(cls, parser):
        defaults = cls.http_defaults() if parser.prog.endswith(" server") else cls()
        group = parser.add_argument_group("shared inference settings")
        group.add_argument(
            "--config", help="Optional JSON overrides; omit to use built-in defaults"
        )
        # Suppressed defaults distinguish explicit CLI values from file defaults.
        for flags, name, kind in (
            (("--max-tokens",), "max_tokens", _optional_int),
            (("--max-context-length",), "max_context_length", _optional_int),
            (("--temperature",), "temperature", float),
            (("--top-p",), "top_p", float),
            (("--top-k",), "top_k", int),
            (("--min-p",), "min_p", float),
        ):
            description = {
                "max_tokens": "Maximum output tokens",
                "max_context_length": "Maximum total input plus output tokens",
            }.get(name, name)
            group.add_argument(
                *flags,
                dest=name,
                type=kind,
                default=argparse.SUPPRESS,
                help=f"{description} (default: {getattr(defaults, name)})",
            )
        group.add_argument(
            "--enable-thinking",
            dest="enable_thinking",
            action=argparse.BooleanOptionalAction,
            default=argparse.SUPPRESS,
            help=f"Enable/disable thinking (default: {defaults.enable_thinking})",
        )
        group.add_argument(
            "--reasoning-effort",
            choices=("low", "high"),
            default=argparse.SUPPRESS,
            help=f"Thinking effort (default: {defaults.reasoning_effort})",
        )
        group.add_argument(
            "--chat-template-kwargs",
            dest="chat_template_kwargs",
            type=_json_object,
            default=argparse.SUPPRESS,
            help="Additional chat-template arguments as a JSON object",
        )
        for name, kind in (
            ("seed", _optional_int),
            ("draft_temperature", float),
            ("draft_top_p", float),
            ("draft_top_k", int),
            ("max_draft_tokens", int),
            ("confidence_threshold", float),
            ("cache_limit_mib", _optional_int),
        ):
            group.add_argument(
                "--" + name.replace("_", "-"), type=kind,
                default=argparse.SUPPRESS,
                help=f"{name} (default: {getattr(defaults, name)})",
            )
        for name in ("asynchronous_draft", "async_context", "compile_draft_chain", "apply_chat_template", "stop_at_eos", "clear_cache"):
            group.add_argument(
                "--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                default=argparse.SUPPRESS,
                help=f"{name} (default: {getattr(defaults, name)})",
            )
        cls.add_kv_arguments(parser, suppress_defaults=True)

    @classmethod
    def from_args(cls, parser, args):
        try:
            values = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
            options = vars(args)
            server = getattr(args, "command", None) == "server"
            if server:
                cls.reject_http_kv_options(values)
                cls.reject_http_kv_options(options)
            base = cls.http_defaults() if server else cls()
            config = base.overlay(values)
            return config.overlay({item.name: options[item.name] for item in fields(cls) if item.name in options})
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))

    @classmethod
    def add_kv_arguments(cls, parser, *, suppress_defaults=False):
        """Register only KV/prefill flags for benchmarks and dataset runners."""
        server = parser.prog.endswith(" server")
        defaults = cls.http_defaults() if server else cls()
        group = parser.add_argument_group(
            "prefill settings" if server else "Spark KV and prefill settings"
        )
        group.add_argument(
            "--prefill-step-size",
            type=_positive_int,
            default=argparse.SUPPRESS
            if suppress_defaults
            else defaults.prefill_step_size,
            help=f"Prompt chunk size (default: {defaults.prefill_step_size})",
        )
        group.add_argument(
            "--kv-bits",
            type=_kv_bits,
            choices=(None, 4, 8),
            metavar="{none,4,8}",
            default=argparse.SUPPRESS if suppress_defaults else defaults.kv_bits,
            help=(argparse.SUPPRESS if server else
                  "KV quantization bits (default: none); 'none' keeps model precision"),
        )
        group.add_argument(
            "--kv-group-size",
            type=int,
            choices=(32, 64, 128),
            default=argparse.SUPPRESS if suppress_defaults else defaults.kv_group_size,
            help=argparse.SUPPRESS if server else None,
        )
        group.add_argument(
            "--quantized-kv-start",
            type=_nonnegative_int,
            default=argparse.SUPPRESS
            if suppress_defaults
            else defaults.quantized_kv_start,
            help=argparse.SUPPRESS if server else None,
        )
        group.add_argument(
            "--kv-quantization-scope",
            choices=("all", "full-attention"),
            default=(
                argparse.SUPPRESS
                if suppress_defaults
                else defaults.kv_quantization_scope
            ),
            help=(argparse.SUPPRESS if server else
                  "When KV quantization is enabled, default to full-attention layers only"),
        )

    def kv_kwargs(self):
        """KV/prefill fields from the same generation configuration."""
        return {name: getattr(self, name) for name in _KV_FIELDS}



def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _optional_int(value):
    return None if value.lower() == "none" else int(value)


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def _kv_bits(value):
    return None if value.lower() == "none" else int(value)
