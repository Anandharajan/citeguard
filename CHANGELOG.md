# Changelog

## 0.1.1 (2026-11-24)
- First release archived on Zenodo with a DOI; no functional changes since 0.1.0.

## 0.1.0 (2026-11-24)
- Claim-level support checking with a local DeBERTa NLI model, rationale selection, and a contradicted label.
- SciFact benchmark (train-tuned thresholds, dev-reported results).
- Crossref + OpenAlex resolution (OpenAlex title search for venues Crossref lacks; abstracts from OpenAlex).
- Retraction / preprint-version / metadata checks, HTML report, CLI, local MCP server.
- Fixes: scikit-learn dependency; UTF-8 console output on Windows; decimal-safe sentence splitting.
