"""A3: version drift: preprint -> published, retractions / expressions of concern, bib diffs."""
from __future__ import annotations

from .matching import Record, jaccard, surnames, title_sim

RETRACTION_TYPES = {"retraction", "expression-of-concern", "expression_of_concern", "withdrawal", "removal"}
CORRECTION_TYPES = {"correction", "erratum", "corrigendum", "addendum"}


def published_version(pre: Record, candidates: list[Record]) -> Record | None:
    """Preprint -> published if title similarity >= 0.92, author Jaccard >= 0.6 and later-or-same year."""
    for doi in pre.relations.get("is-preprint-of", []):
        for c in candidates:
            if c.doi and c.doi.lower() == doi.lower():
                return c
    best = None
    for c in candidates:
        if c.is_preprint:
            continue
        if (title_sim(pre.title, c.title) >= 0.92 and jaccard(surnames(pre.authors), surnames(c.authors)) >= 0.6
                and (pre.year is None or c.year is None or c.year >= pre.year)):
            if best is None or title_sim(pre.title, c.title) > title_sim(pre.title, best.title):
                best = c
    return best


def drift_flags(rec: Record) -> list[dict[str, str]]:
    flags = []
    for u in rec.updates:
        t = u.get("type", "").lower()
        if t in RETRACTION_TYPES:
            flags.append({"flag": "retracted" if "retraction" in t or "withdraw" in t or "removal" in t
                          else "expression of concern", "notice": u.get("DOI", "")})
        elif t in CORRECTION_TYPES:
            flags.append({"flag": "corrected", "notice": u.get("DOI", "")})
    if rec.is_preprint:
        pub = published_version(rec, rec.related_published)
        if pub:
            flags.append({"flag": "newer version", "notice": pub.doi or pub.title})
    return flags
