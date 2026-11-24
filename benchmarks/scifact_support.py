"""Claim-support benchmark on SciFact (Wadden et al., EMNLP 2020).

Each item is (claim, cited abstract) labelled SUPPORT, CONTRADICT or NEI (cited, but no
rationale). This is the abstract-only setting of CiteGuard's claim-support check.

Protocol (fixed; no tuning on dev):
  * thresholds tuned on the SciFact *train* split, metrics reported on the *dev* split
    (SciFact test labels are not public);
  * scorers: lexical fallback (no model) vs local cross-encoder NLI (DeBERTa-v3-small);
  * metrics: 3-way macro-F1, per-class F1, and binary SUPPORT-vs-rest AUROC.

The dataset is downloaded at run time into benchmarks/data/ (git-ignored) and is not
redistributed with CiteGuard.
"""
from __future__ import annotations

import csv
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from citeguard.algorithms.support import (  # noqa: E402
    CrossEncoderNLI, aggregate, calibrate_3way, classify, lexical_nli, macro_f1, score_sentences,
)

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA = HERE / "data" / "scifact"
OUT = HERE / "results"


def fetch() -> None:
    if (DATA / "claims_dev.jsonl").exists():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    arc = DATA / "data.tar.gz"
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, arc)
    with tarfile.open(arc) as t:
        for m in t.getmembers():
            name = Path(m.name).name
            if name in ("claims_train.jsonl", "claims_dev.jsonl", "corpus.jsonl") and m.isfile() and not name.startswith("._"):
                m.name = name
                t.extract(m, DATA, filter="data")
    arc.unlink()


def load(split: str) -> list[tuple[str, list[str], str]]:
    corpus = {}
    for line in open(DATA / "corpus.jsonl", encoding="utf-8"):
        d = json.loads(line)
        corpus[d["doc_id"]] = d["abstract"]
    items = []
    for line in open(DATA / f"claims_{split}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        for doc in d["cited_doc_ids"]:
            ev = d["evidence"].get(str(doc))
            items.append((d["claim"], corpus[int(doc)], ev[0]["label"] if ev else "NEI"))
    return items


def sentence_probs(items, nli) -> list[list[list[float]]]:
    return [[list(p) for p in score_sentences(claim, sents, nli)] for claim, sents, _ in items]


def scores(items, probs) -> tuple[np.ndarray, np.ndarray]:
    agg = [aggregate(claim, sents, [tuple(p) for p in pr]) for (claim, sents, _), pr in zip(items, probs)]
    return np.array([a[0] for a in agg]), np.array([a[1] for a in agg])


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    fetch()
    train, dev = load("train"), load("dev")
    print(f"train pairs {len(train)}, dev pairs {len(dev)}")
    scorers = {"lexical (no model)": lexical_nli}
    try:
        scorers["cross-encoder NLI (DeBERTa-v3-small)"] = CrossEncoderNLI()
    except ImportError:
        print('sentence-transformers missing: pip install "citeguard[nli]"')
    rows = []
    for name, nli in scorers.items():
        cache = OUT / f"probs_{'lexical' if nli is lexical_nli else 'cross_encoder'}.json"
        t0 = time.perf_counter()
        if cache.exists():  # per-sentence model scores are deterministic: cache them, aggregation stays cheap
            pt, pd = json.loads(cache.read_text())
        else:
            pt, pd = sentence_probs(train, nli), sentence_probs(dev, nli)
            OUT.mkdir(exist_ok=True)
            cache.write_text(json.dumps([pt, pd]))
        st, ct = scores(train, pt)
        sd, cd = scores(dev, pd)
        secs = time.perf_counter() - t0
        th, train_f1 = calibrate_3way(list(st), list(ct), [x[2] for x in train], grid=97)
        gold = [x[2] for x in dev]
        pred = []
        for s, c in zip(sd, cd):
            lab = classify(s, c, th)
            pred.append("SUPPORT" if lab == "supported" else "CONTRADICT" if lab == "contradicted" else "NEI")
        per = {c: macro_f1(pred, gold, (c,)) for c in ("SUPPORT", "CONTRADICT", "NEI")}
        y = np.array([g == "SUPPORT" for g in gold], int)
        rows.append({"scorer": name, "dev_pairs": len(dev), "macro_f1": round(macro_f1(pred, gold, ("SUPPORT", "CONTRADICT", "NEI")), 3),
                     "f1_support": round(per["SUPPORT"], 3), "f1_contradict": round(per["CONTRADICT"], 3),
                     "f1_nei": round(per["NEI"], 3), "auroc_support": round(auroc(y, sd), 3),
                     "train_macro_f1": round(train_f1, 3), "t_supported": round(th.supported, 3),
                     "t_contradicted": round(th.contradicted, 3), "seconds_train_plus_dev": round(secs, 1)})
        print(rows[-1])
    OUT.mkdir(exist_ok=True)
    with open(OUT / "scifact_support.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
