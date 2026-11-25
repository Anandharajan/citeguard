"""A1: field-level matching of a BibTeX entry to candidate canonical records, with isotonic
calibration to P(real and correctly described). Existence checking is NOT the contribution;
this layer only feeds the support and drift checks."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Record:
    """Canonical metadata from Crossref / OpenAlex / Semantic Scholar."""
    title: str
    authors: list[str]
    year: int | None
    venue: str = ""
    doi: str | None = None
    source: str = ""
    abstract: str = ""
    fulltext: str = ""
    is_preprint: bool = False
    relations: dict[str, list[str]] = field(default_factory=dict)  # e.g. {"is-preprint-of": [doi]}
    updates: list[dict[str, str]] = field(default_factory=list)  # Crossref update-to: [{"type": "retraction", "DOI": ...}]
    related_published: list[Record] = field(default_factory=list)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def title_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def surnames(authors: list[str]) -> set[str]:
    out = set()
    for a in authors:
        if norm(a) in ("others", "et al"):
            continue
        a = norm(a.split(",")[0]) if "," in a else norm(a).split(" ")[-1] if norm(a) else ""
        if a:
            out.add(a.split(" ")[-1])
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


W = {"title": 0.55, "authors": 0.25, "year": 0.1, "venue": 0.1}


def raw_score(title: str, authors: list[str], year: int | None, venue: str, rec: Record) -> dict[str, float]:
    f = {"title": title_sim(title, rec.title), "authors": jaccard(surnames(authors), surnames(rec.authors)),
         "year": float(year is not None and rec.year is not None and abs(year - rec.year) <= 1),
         "venue": title_sim(venue, rec.venue) if venue and rec.venue else 0.5}
    f["score"] = sum(W[k] * f[k] for k in W)
    return f


class Calibrator:
    """Isotonic map raw score -> P(real & correctly described), fitted on a labelled set."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        self.iso.fit(scores, labels)
        self.fitted = True
        return self

    def __call__(self, s: float) -> float:
        return float(self.iso.predict([s])[0]) if self.fitted else s


def mismatches(title: str, authors: list[str], year: int | None, rec: Record) -> list[str]:
    out = []
    if title_sim(title, rec.title) < 0.95:
        out.append(f"title differs: canonical '{rec.title}'")
    missing = surnames(rec.authors) - surnames(authors)
    if missing and not any(norm(a) in ("others", "et al") for a in authors):
        out.append(f"authors missing from bib: {', '.join(sorted(missing))}")
    extra = surnames(authors) - surnames(rec.authors)
    if extra:
        out.append(f"authors not in canonical record: {', '.join(sorted(extra))}")
    if year and rec.year and year != rec.year:
        out.append(f"year {year} vs canonical {rec.year}" +
                   (" (check preprint versus published version)" if rec.is_preprint else ""))
    return out
