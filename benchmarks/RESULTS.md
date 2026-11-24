# CiteGuard: results and honest status

All numbers below come from the automated test suite (`python -m pytest`) on **synthetic data**,
using fixed seeds. None of them is evidence from real users or real institutional data yet.

| Check | Data | Result |
|---|---|---|
| End-to-end badges on a fixture paper | 4 entries + 5 citations (recorded fixtures) | all expected badges and labels |
| 200-entry bib, warm cache | fixtures | < 60 s |

## Gaps and caveats (read before writing the paper)

- The default NLI is a transparent lexical fallback (`lexical_nli`). Install sentence-transformers and use `nli_cross_encoder()` for any reported claim-support numbers.
- The ~1,000-pair claim–citation dataset, comparison against CheckIfExist/RefChecker, and the Overleaf side panel are not built.
- Semantic Scholar client not added (Crossref + OpenAlex only).
