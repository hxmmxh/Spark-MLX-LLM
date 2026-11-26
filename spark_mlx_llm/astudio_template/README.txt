HTTP startup prefixes

The default startup plan warms title.txt, followed by three cumulative cuts of
template.txt: System 1, System 1 + System 2, and those systems + fixed User 1.
The last main cut ends at </INSTRUCTIONS><environment_context>; the title ends
at <environment_context>. Environment contents, the final question, and the
assistant generation prefix are supplied by the client.

These assets come from the Vulkan staged-prefill snapshot dated 2026-11-26.
They were re-rendered byte-for-byte with the MLX chat template. The source
revision, hashes, byte cuts and verified token counts are in template-info.json.
For Spark-X2.6-4B-INT4, main inputs have 26,320 / 34,795 / 36,451 tokens;
the independent title prefix has 2,388. Each checkpoint leaves its last token
for the next request's logits. The two later main stages reuse the preceding
stage's target/draft state instead of computing the whole input again.

The service checks both raw asset hashes and byte boundaries, normalizes to LF
by default, encodes with add_special_tokens=False, and verifies that main token
sequences are nested. It does not apply a second chat template or sample output.

--single-prefill warms only the full bundled main prefix; --prefill-file FILE
uses one custom rendered prefix; --skip-prefill disables startup warmup.
--prefill-newlines preserve retains original newlines. These are startup options.
All stored checkpoints share --prompt-cache-size / --prompt-cache-bytes with
request and generated KV. Insufficient capacity can evict a startup checkpoint.
Recapture after tool, skill or fixed workspace-instruction changes.
See docs/inference.md for the complete startup and cache behavior.
