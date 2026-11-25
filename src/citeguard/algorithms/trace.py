"""Experimental TRACE-NLI: conservative quantitative and citation-group screening."""
from __future__ import annotations

import re
from itertools import groupby
from typing import Any

from .support import NLI, lexical_nli, support, split_sentences

# Deliberately narrow grammar: compare only matching metric, direction and context.
_NUM = re.compile(
    r"\b(?P<metric>[\w -]{2,60}?)\s+(?:was\s+|were\s+)?"
    r"(?P<direction>decreased|increased|reduced|improved|fell|rose)\s+by\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>percentage points|%|milliseconds|ms|seconds|s)(?!\w)",
    re.I,
)


def quantity(text: str) -> dict[str, Any] | None:
    text = text.replace(r"\%", "%")
    if re.search(r"\b(about|approximately|between|at least|at most|up to|not|never|if|unless)\b|[<>±~]", text, re.I):
        return None
    matches = list(_NUM.finditer(text))
    if len(matches) != 1:
        return None
    m = matches[0]
    unit = m["unit"].lower()
    value = float(m["value"])
    if unit in ("seconds", "s", "milliseconds", "ms"):
        value *= 1000 if unit in ("seconds", "s") else 1
        unit = "ms"
    direction = "down" if m["direction"].lower() in ("decreased", "reduced", "fell") else m["direction"].lower()
    if direction in ("increased", "rose"):
        direction = "up"
    return {"metric": " ".join(m["metric"].lower().split()).removeprefix("the "),
            "direction": direction, "value": value, "unit": unit,
            "context": text[m.end():].strip(" .;").lower()}


def quantitative(claim: str, evidence: str) -> dict[str, Any]:
    a, b = quantity(claim), quantity(evidence)
    if a is None or b is None:
        return {"status": "not comparable"}
    if any(a[k] != b[k] for k in ("metric", "unit", "context")):
        return {"status": "not comparable", "claim": a, "evidence": b}
    same = abs(a["value"] - b["value"]) <= 1e-9 and a["direction"] == b["direction"]
    return {"status": "match" if same else "potential mismatch", "claim": a, "evidence": b}


def atomic_claims(claim: str) -> list[str]:
    """Split explicit independent clauses only; do not invent missing subjects."""
    parts = re.split(r";\s+|\s+and\s+(?=(?:the|a|an|our|their|this|these)\s+\w+\s+(?:is|are|was|were|reduced|increased|decreased|improved|shows?)\b)", claim, flags=re.I)
    if len(parts) > 1 and all(len(p.split()) >= 4 for p in parts):
        return [p.strip(" ;") for p in parts]
    return [claim]


def assess(claim: str, source: str, nli: NLI = lexical_nli, fulltext: str = "") -> dict[str, Any]:
    """Selective heuristic routing; scores are not calibrated probabilities."""
    result = support(claim, source, nli)
    route = "abstract"
    initial_checks = [quantitative(claim, s) for _, s in split_sentences(source)]
    needs_quantity = quantity(claim) is not None and not any(c["status"] == "match" for c in initial_checks)
    if fulltext and (result["label"] != "supported" or needs_quantity):
        result = support(claim, fulltext, nli)
        route = "provided full text"
    checks = [dict(quantitative(claim, sentence), sentence=sentence, offset=offset)
              for offset, sentence in split_sentences(fulltext if route == "provided full text" else source)]
    checks = [c for c in checks if c["status"] != "not comparable"]
    statuses = {c["status"] for c in checks}
    label = result["label"]
    if "potential mismatch" in statuses:
        label = "requires review"
    elif quantity(claim) and "match" not in statuses:
        label = "insufficient quantitative evidence"
    elif label in ("unsupported", "cannot verify"):
        label = "insufficient accessible evidence"
    elif nli is lexical_nli or (label == "supported" and (result.get("score") or 0) < 0.5):
        label = "requires review"
    return {**result, "label": label, "route": route, "quantitative": checks,
            "calibrated": False}


def evidence_graph(claims: list[dict[str, Any]], sources: dict[str, tuple[str, str]],
                   nli: NLI = lexical_nli) -> list[dict[str, Any]]:
    """Group co-citations by line and claim; preserve individual evidence edges."""
    groups = []
    for (line, claim), rows in groupby(claims, key=lambda c: (c["line"], c["claim"])):
        keys = list(dict.fromkeys(c["key"] for c in rows))
        atoms = atomic_claims(claim)
        edges = []
        for i, atom in enumerate(atoms):
            for key in keys:
                abstract, fulltext = sources.get(key, ("", ""))
                edges.append({"atom": i, "key": key, **assess(atom, abstract, nli, fulltext)})
        covered = {e["atom"] for e in edges if e["label"] == "supported"}
        conflicts = [e for e in edges if e["label"] in ("contradicted", "requires review")]
        groups.append({"line": line, "claim": claim, "atoms": atoms, "keys": keys,
                       "edges": edges, "coverage": len(covered) / len(atoms),
                       "label": "requires review" if conflicts else
                       "supported" if len(covered) == len(atoms) else
                       "partially supported" if covered else "insufficient accessible evidence"})
    return groups
