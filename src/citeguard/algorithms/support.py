"""A2: claim-source entailment scoring (CSE), max-then-calibrate.

e_j = P_entail(claim | s_j) - P_contra(claim | s_j) for each source sentence s_j;
score = max_j e_j, classified by calibrated thresholds; abstain ("cannot verify") when the source
has fewer than 3 sentences. Returns the top-2 supporting sentences with character offsets.

The NLI scorer is pluggable. `nli_cross_encoder()` loads a local cross-encoder
(cross-encoder/nli-deberta-v3-small) when sentence-transformers is installed. The default
`lexical_nli` is a transparent fallback (content-word coverage minus a negation/antonym
contradiction cue) used in tests and when no model is available; report which scorer was used.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Callable

NLI = Callable[[str, str], tuple[float, float]]  # (premise, hypothesis) -> (P_entail, P_contra)

_STOP = set("""a an the of and or to in on for with by is are was were be been as at from this that these those
it its into we our their they them which while such can could may might also than then thus has have had not no
using use used show shows shown study paper work results result approach method methods based""".split())
_NEG = re.compile(r"\b(no|not|never|fails?|failed|cannot|unable|without|neither|nor|worse|decreas\w*|lower|less)\b")


def content(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", s.lower())
    return [t[:-1] if t.endswith("s") and len(t) > 4 else t for t in toks if t not in _STOP and len(t) > 2]


def lexical_nli(premise: str, hypothesis: str) -> tuple[float, float]:
    h, p = content(hypothesis), set(content(premise))
    if not h:
        return 0.0, 0.0
    cov = sum(t in p for t in h) / len(h)
    neg_mismatch = bool(_NEG.search(premise.lower())) != bool(_NEG.search(hypothesis.lower()))
    if neg_mismatch and cov >= 0.5:
        return 0.1 * cov, cov
    return cov, 0.0


DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"


class CrossEncoderNLI:
    """Local NLI cross-encoder (sentence-transformers). Install with: pip install "citeguard[nli]".

    The model runs on this machine; the manuscript text never leaves it. Scores are softmax
    probabilities read from the model's own id2label mapping (no hard-coded label order).
    """

    def __init__(self, model: str = DEFAULT_NLI_MODEL, device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder
        self.name = model
        self.ce = CrossEncoder(model, device=device)
        labels = {str(v).lower(): int(k) for k, v in self.ce.model.config.id2label.items()}
        self.ie, self.ic = labels["entailment"], labels["contradiction"]
        self.batch_size = batch_size

    def batch(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        import numpy as np
        if not pairs:
            return []
        logits = np.asarray(self.ce.predict(pairs, batch_size=self.batch_size, show_progress_bar=False), float)
        logits = np.atleast_2d(logits)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return [(float(r[self.ie]), float(r[self.ic])) for r in p]

    def __call__(self, premise: str, hypothesis: str) -> tuple[float, float]:
        return self.batch([(premise, hypothesis)])[0]


def nli_cross_encoder(model: str = DEFAULT_NLI_MODEL) -> CrossEncoderNLI:  # pragma: no cover - loads a model
    return CrossEncoderNLI(model)


def split_sentences(text: str) -> list[tuple[int, str]]:
    out = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text):
        s = m.group().strip()
        if len(s.split()) >= 4:
            out.append((m.start() + (len(m.group()) - len(m.group().lstrip())), s))
    return out


@dataclass
class Thresholds:
    supported: float = 0.6
    partial: float = 0.3
    contradicted: float = 0.5


# Per-backend defaults. The cross-encoder values are tuned on the SciFact *train* split only
# (benchmarks/scifact_support.py); results are reported on the dev split.
LEXICAL_THRESHOLDS = Thresholds(0.6, 0.3, 0.5)
CROSS_ENCODER_THRESHOLDS = Thresholds(0.6, 0.3, 0.5)


def score_sentences(claim: str, sents: list[str], nli: NLI) -> list[tuple[float, float]]:
    batch = getattr(nli, "batch", None)
    if callable(batch):
        return list(batch([(s, claim) for s in sents]))
    return [nli(s, claim) for s in sents]


def classify(best_support: float, best_contra: float, th: Thresholds) -> str:
    if best_support >= th.supported and best_support >= best_contra:
        return "supported"
    if best_contra >= th.contradicted:
        return "contradicted"
    if best_support >= th.supported:
        return "supported"
    if best_support >= th.partial:
        return "partially supported"
    return "unsupported"


def support(claim: str, source_text: str | list[str], nli: NLI = lexical_nli,
            th: Thresholds | None = None) -> dict[str, object]:
    """source_text: raw text (split into sentences) or an already-split list of sentences."""
    if th is None:
        th = CROSS_ENCODER_THRESHOLDS if isinstance(nli, CrossEncoderNLI) else LEXICAL_THRESHOLDS
    if isinstance(source_text, list):
        offs, pos = [], 0
        for s in source_text:
            offs.append(pos)
            pos += len(s) + 1
        sents = [(o, s) for o, s in zip(offs, source_text) if len(s.split()) >= 4]
    else:
        sents = split_sentences(source_text)
    if len(sents) < 3:
        return {"label": "cannot verify", "score": None, "contradiction": None, "evidence": [],
                "reason": "fewer than 3 source sentences"}
    probs = score_sentences(claim, [s for _, s in sents], nli)
    sup = [(pe - pc, off, s) for (pe, pc), (off, s) in zip(probs, sents)]
    con = max(pc - pe for pe, pc in probs)
    sup.sort(key=lambda t: -t[0])
    best = sup[0][0]
    return {"label": classify(best, con, th), "score": round(best, 3), "contradiction": round(con, 3),
            "evidence": [{"offset": o, "sentence": s, "score": round(e, 3)} for e, o, s in sup[:2]]}


def calibrate_3way(support_scores: list[float], contra_scores: list[float], labels: list[str],
                   grid: int = 41) -> tuple[Thresholds, float]:
    """Tune (supported, contradicted) cut-points for macro-F1 over SUPPORT / CONTRADICT / NEI.
    'partially supported' and 'unsupported' both count as NEI. Use on a training split only."""
    import numpy as np
    cands = np.linspace(-0.2, 0.995, grid)
    best, bt = -1.0, Thresholds()
    for ts in cands:
        for tc in cands:
            th = Thresholds(float(ts), float(ts) / 2, float(tc))
            pred = []
            for s, c in zip(support_scores, contra_scores):
                lab = classify(s, c, th)
                pred.append("SUPPORT" if lab == "supported" else "CONTRADICT" if lab == "contradicted" else "NEI")
            f = macro_f1(pred, labels, ("SUPPORT", "CONTRADICT", "NEI"))
            if f > best:
                best, bt = f, th
    return bt, best


def macro_f1(pred: list[str], gold: list[str], classes: tuple[str, ...]) -> float:
    f1s = []
    for c in classes:
        tp = sum(p == c and y == c for p, y in zip(pred, gold))
        fp = sum(p == c and y != c for p, y in zip(pred, gold))
        fn = sum(p != c and y == c for p, y in zip(pred, gold))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return sum(f1s) / len(f1s)


def calibrate_thresholds(scores: list[float], labels: list[str]) -> Thresholds:
    """Choose the two cut-points maximising macro-F1 over supported / partially / unsupported."""
    import itertools
    cands = sorted(set(round(s, 3) for s in scores))
    best, bt = -1.0, Thresholds()
    for lo, hi in itertools.combinations(cands, 2):
        pred = ["supported" if s >= hi else "partially supported" if s >= lo else "unsupported" for s in scores]
        f = macro_f1(pred, labels, ("supported", "partially supported", "unsupported"))
        if f > best:
            best, bt = f, Thresholds(hi, lo)
    return bt
