# CiteGuard

**Does the paper you cite actually say what you claim it says?** CiteGuard checks every `\cite` in your
LaTeX paper against the cited work, using a local language model, and flags retracted papers, newer
published versions and wrong BibTeX metadata — before a reviewer does.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24502492.svg)](https://doi.org/10.5281/zenodo.24502492) [![CI](https://github.com/Anandharajan/citeguard/actions/workflows/ci.yml/badge.svg)](https://github.com/Anandharajan/citeguard/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anandharajan/citeguard/blob/main/notebooks/quickstart.ipynb) [![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-black?logo=github)](https://codespaces.new/Anandharajan/citeguard) [![GitHub stars](https://img.shields.io/github/stars/Anandharajan/citeguard?style=social)](https://github.com/Anandharajan/citeguard/stargazers)

> ⭐ **If CiteGuard catches even one bad citation for you, please [star the repo](https://github.com/Anandharajan/citeguard)** — it helps other researchers find it.

## Why

LLM-assisted writing produces citations that *exist* but do not support the sentence they are attached to,
and plain existence checkers miss that. CiteGuard focuses on what they miss:

| Badge | Meaning |
|---|---|
| ✔ | reference found, metadata matches |
| ◐ | the cited work does **not support** (or **contradicts**) the citing sentence |
| ↻ | a preprint you cite now has a **published version** |
| ⛔ | the cited work was **retracted** or has an expression of concern |
| ⚠ | title / authors / year differ from the canonical record |
| ✖ | not found in Crossref / OpenAlex |

## Try CiteGuard in your browser

- **No local install (recommended for researchers):** [open the Colab notebook](https://colab.research.google.com/github/Anandharajan/citeguard/blob/main/notebooks/quickstart.ipynb). Run **Setup**, then **Module 2: Check your own paper**. Select one `.tex` manuscript and one `.bib` bibliography (any filenames). The report appears in the notebook and downloads automatically. Module 1 is an optional sample demo. First run may take several minutes to install dependencies and download the model.
- **For developers, in the browser with VS Code:** [open in GitHub Codespaces](https://codespaces.new/Anandharajan/citeguard).
- **On your machine:**

```bash
pip install "citeguard[nli] @ git+https://github.com/Anandharajan/citeguard"
citeguard check main.tex refs.bib --report report.html
```

Check a single claim against an abstract, fully offline:

```bash
citeguard claim "Residual routing reduces packet latency in mesh networks." abstract.txt
```

Real output with the local NLI model (three claims against the same abstract):

```
supported      score 0.99  | Residual routing reduces packet latency in mesh networks.
contradicted   score 1.00  | Residual routing increases packet latency in mesh networks.
unsupported    score 0.06  | Residual routing improves battery life of phones.
```

## Use it from your AI assistant (MCP)

citeguard ships a local [Model Context Protocol](https://modelcontextprotocol.io) server (stdio only, nothing leaves
your machine). Install with `pip install "citeguard[mcp,nli] @ git+https://github.com/Anandharajan/citeguard"`, then add this to
your client's MCP config (Claude Desktop, Cursor, VS Code and others use the same shape):

```json
{
  "mcpServers": {
    "citeguard": { "command": "citeguard-mcp" }
  }
}
```

Every tool call is written to a local audit log, and you can limit file access to chosen folders with the
`CITEGUARD_ROOTS` environment variable.

## How it works

1. **Parse** `.tex` (citing sentence + context) and `.bib`.
2. **Resolve** each entry on Crossref and OpenAlex (titles/DOIs only; cached, rate-limited) and score the metadata match.
3. **Claim support:** split the cited abstract into sentences and score each against the citing sentence with a local
   NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`). Contradiction is judged only on the 2 sentences most relevant
   to the claim (rationale selection), because small NLI models report spurious contradictions for unrelated sentences.
   Labels: `supported` / `partially supported` / `contradicted` / `unsupported`, with the two best evidence sentences.
4. **Drift:** preprint → published version, retractions and corrections from Crossref update records.

## Experimental TRACE-NLI screening

Project author: Anandharajan TRV (see CITATION.cff). AI-assisted implementation.

The standard manuscript check now includes an additional experimental report:
- Quantitative screening compares explicitly aligned metric, direction, units and context; seconds and milliseconds are normalized.
- Conservative atomic clause splitting and a citation-group evidence graph expose per-reference evidence and coverage.
- Selective routing uses supplied full text when abstract evidence is insufficient; automatic full-text retrieval is not implemented.
- Missing evidence and numeric discrepancies are flagged for human review. Scores are not calibrated probabilities.

The original NLI results remain available separately. TRACE-NLI is a research prototype, not a proven novel algorithm or independently benchmarked improvement. Existing SciFact numbers below are for the original scorer, not TRACE-NLI. Run `pytest tests/test_trace.py` for regression checks.

## Results (SciFact, honest numbers)

Claim–abstract pairs from [SciFact](https://github.com/allenai/scifact) (Wadden et al., EMNLP 2020); thresholds tuned on
the train split, reported on the dev split (340 pairs). Reproduce with `python benchmarks/scifact_support.py`.

| Scorer | 3-way macro-F1 | F1 supports | F1 contradicts | AUROC (supports vs rest) |
|---|---|---|---|---|
| Lexical fallback (no model) | 0.507 | 0.636 | 0.2 | 0.758 |
| **Local NLI (DeBERTa-v3-small) + rationale selection** | **0.614** | **0.667** | **0.533** | **0.836** |

These are abstract-only results with a small CPU model: treat a ◐ badge as "go and read this paper", not as a verdict.
Full-text checking and larger models are on the roadmap.

## Privacy

Only titles and DOIs are sent to Crossref/OpenAlex for metadata resolution. With the local CLI, manuscript text stays on your machine and the NLI model runs locally. Colab and Codespaces process manuscripts on remote cloud infrastructure; use the local CLI for confidential or unpublished material. In Colab, delete the runtime when finished. Set `CITEGUARD_MAILTO` to identify yourself to the Crossref/OpenAlex polite pools.

## Roadmap

Overleaf side panel · full-text (open-access) support checking · a released claim–citation dataset · comparison with
existing citation checkers.

## Contributing

Bug reports, ideas and pull requests are welcome: [open an issue](https://github.com/Anandharajan/citeguard/issues).
Run `python -m pytest -q` before sending a pull request.

## Citation

If you use citeguard in research, please cite it (see [CITATION.cff](CITATION.cff), or GitHub's *Cite this repository* button).

## Licence

MIT © 2026 Anandharajan TRV, Department of ECE, St. Joseph College of Engineering, Chennai.

---

**If citeguard saved you time, please ⭐ [star the repository](https://github.com/Anandharajan/citeguard/stargazers)** — it is
the simplest way to help other researchers find it.
