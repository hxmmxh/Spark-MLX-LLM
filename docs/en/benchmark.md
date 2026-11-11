# Unified throughput benchmark

Measurements use the same `generate()` and `GenerationConfig` as Python and the generation CLI.
Defaults are built into the code; no configuration file is required. HTTP retains its original service defaults.
See the [unified inference guide](../inference.md) for configuration and metric definitions.

```sh
.venv/bin/spark-mlx benchmark --model TARGET --draft-model DRAFT \
    --input-file gen.txt --trials 5 --warmup-tokens 128 > result.json
```

Omit the draft option for a target-only model. Use `--max-tokens none` for natural EOS,
or `--max-tokens N --no-stop-at-eos` for fixed-length diagnostics; compare these as separate protocols.

Warmup only shortens the output limit. Sampling, seed, cache and async settings remain the same.
Every planned repetition is retained, including slow runs. Each records tokens, stop reason, times,
prompt hash and effective settings. Pooled decode throughput is
`sum(output tokens - 1) / sum(generation_seconds)`; the first token belongs to prefill and EOS counts as output.
The report also includes min/max/median and exact output repeatability.

CPU conformance verifies interface behavior. Formal GPU baselines additionally follow the
[experiment framework](../../experiments/README.md), freezing current source/binaries and recording environment evidence.
