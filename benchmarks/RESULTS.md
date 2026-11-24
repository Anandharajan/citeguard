# CiteGuard: results and honest status

## Claim support on SciFact (`benchmarks/scifact_support.py`)

[SciFact](https://github.com/allenai/scifact) (Wadden et al., EMNLP 2020): scientific claims paired with the abstract
they cite, labelled SUPPORT / CONTRADICT / NEI. Abstract-only setting. Thresholds tuned on the **train** split (919
pairs); metrics on the **dev** split (340 pairs). SciFact test labels are not public. The dataset is downloaded at run
time and not redistributed.

| Scorer | 3-way macro-F1 | F1 SUPPORT | F1 CONTRADICT | F1 NEI | AUROC (SUPPORT vs rest) |
|---|---|---|---|---|---|
| Lexical fallback (no model) | 0.507 | 0.636 | 0.200 | 0.685 | 0.758 |
| DeBERTa-v3-small NLI, contradiction over all sentences | 0.546 | 0.595 | 0.450 | 0.592 | 0.836 |
| **DeBERTa-v3-small NLI + rationale selection (k = 2), shipped** | **0.614** | **0.667** | **0.533** | **0.642** | **0.836** |

Shipped thresholds (train-tuned): supported ≥ 0.012, contradicted ≥ 0.97 (entailment/contradiction margins).

### How the method changed, and why

1. The first run took the maximum contradiction over every abstract sentence. A real example (the Transformer paper)
   showed that the small NLI model reports ~1.0 "contradiction" for unrelated background sentences, so we added
   rationale selection: contradiction is judged only on the k = 2 sentences most relevant to the claim. k = 2 was fixed
   before the re-run.
2. Exploratory sweep afterwards (per-sentence scores cached), thresholds always tuned on train:

| k (rationale sentences) | train macro-F1 | dev macro-F1 |
|---|---|---|
| 1 | 0.575 | 0.586 |
| **2 (pre-specified, shipped)** | 0.572 | **0.614** |
| 3 | 0.571 | 0.597 |
| all sentences | 0.548 | 0.546 |

   Train scores for k = 1/2/3 are within noise. Train alone would have picked k = 1 (dev 0.586), so the shipped
   0.614 should be read with that in mind.

## Known limitations

- Small CPU model, abstracts only: paraphrases can be missed (e.g. "replaced recurrence" vs "dispensing with
  recurrence"). A ◐ badge means "read this paper", not a verdict.
- Full-text checking, a larger NLI model, and the planned annotated claim–citation dataset (≈1,000 pairs, 2 annotators)
  are not done. There is no comparison yet with existing checkers (CheckIfExist, RefChecker).
- Existence checking is not claimed as a contribution.

## Unit tests

16 tests, no network (recorded fixtures), including the NLI wrapper with a stand-in model, the 3-way calibration, the
MCP audit log and read-only roots.
