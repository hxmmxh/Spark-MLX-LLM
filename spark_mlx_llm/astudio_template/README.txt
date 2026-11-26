HTTP startup prefix

template.txt is the rendered fixed prefix captured from an actual AStudio
3.7.1 request on 2026-11-26, with Sites skills 1.0.18. See template-info.json
for the capture time, SHA-256 and verified token counts. This snapshot replaces
the older template originally copied from the Vulkan release.

The prefix ends at </INSTRUCTIONS><environment_context>. Environment contents,
the final user question and the assistant generation prefix are excluded.
The service checks the bundled raw template hash, defaults to LF newlines,
and encodes with add_special_tokens=False. No second chat template is applied.
The last input token is left for the request to compute its logits.

Client tool/skill settings can change this prefix; recapture after such changes.
Use --prefill-file FILE for another rendered prefix, --prefill-newlines preserve
for original newlines, or --skip-prefill to skip startup warmup.
See docs/inference.md for ordinary and dSpark cache behavior.
