# Changelog

## 0.1.0 (unreleased)
- Claim-level support checking with a local DeBERTa NLI model, rationale selection, and a contradicted label.
- SciFact benchmark (train-tuned thresholds, dev-reported results).
- Crossref + OpenAlex resolution (OpenAlex title search for venues Crossref lacks; abstracts from OpenAlex).
- Retraction / preprint-version / metadata checks, HTML report, CLI, local MCP server.
- Fixes: scikit-learn dependency; UTF-8 console output on Windows; decimal-safe sentence splitting.
