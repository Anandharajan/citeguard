"""Public API (CLI and MCP). Badges: OK real, MISMATCH metadata, NOT_FOUND, WEAK support,
NEWER version, RETRACTED."""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from . import roots
from .algorithms.drift import drift_flags
from .algorithms.matching import Calibrator, Record, mismatches, raw_score
from .algorithms.support import NLI, CrossEncoderNLI, lexical_nli, support
from .io.clients import Cache, Polite, Resolver
from .io.texbib import BibEntry, citations, parse_bib

BADGE = {"ok": "✔", "mismatch": "⚠", "not_found": "✖", "weak": "◐", "newer": "↻",
         "retracted": "⛔"}


_NLI_CACHE: dict[str, NLI] = {}


def get_nli(kind: str = "auto") -> NLI:
    """'cross-encoder' (local DeBERTa NLI, needs citeguard[nli]), 'lexical' (no model), or 'auto'."""
    if kind == "lexical":
        return lexical_nli
    if kind not in _NLI_CACHE:
        try:
            _NLI_CACHE[kind] = CrossEncoderNLI()
        except ImportError:
            if kind == "cross-encoder":
                raise SystemExit('The NLI model needs: pip install "citeguard[nli]"') from None
            import sys
            print("[citeguard] sentence-transformers not installed: using the lexical fallback scorer. "
                  'For real NLI: pip install "citeguard[nli]"', file=sys.stderr)
            _NLI_CACHE[kind] = lexical_nli
    return _NLI_CACHE[kind]


def _resolver(fetch: Any | None = None) -> Resolver:
    return Resolver(fetch or Polite(cache=Cache()))


def resolve(e: BibEntry, r: Resolver, cal: Calibrator | None = None) -> tuple[Record | None, float]:
    cands = [x for x in [r.by_doi(e.doi)] if x] if e.doi else []
    if not cands and e.title:
        cands = r.by_title(e.title)
    best, bs = None, 0.0
    for c in cands:
        s = raw_score(e.title, e.authors, e.year, e.venue, c)["score"]
        if s > bs:
            best, bs = c, s
    p = (cal or Calibrator())(bs)
    return (best if bs >= 0.6 else None), p


def verify_entry(e: BibEntry, r: Resolver) -> dict[str, Any]:
    rec, p = resolve(e, r)
    out: dict[str, Any] = {"key": e.key, "p_real": round(p, 3), "badges": []}
    if rec is None:
        out["badges"].append("not_found")
        return out
    out["canonical"] = {"title": rec.title, "doi": rec.doi, "year": rec.year, "authors": rec.authors}
    mm = mismatches(e.title, e.authors, e.year, rec)
    out["mismatches"] = mm
    out["badges"].append("mismatch" if mm else "ok")
    for f in drift_flags(rec):
        out["badges"].append("retracted" if f["flag"] in ("retracted", "expression of concern") else
                             "newer" if f["flag"] == "newer version" else "mismatch")
        out.setdefault("drift", []).append(f)
    out["_record"] = rec
    return out


def check(tex: str, bib: str, fetch: Any | None = None, nli: NLI = lexical_nli) -> dict[str, Any]:
    entries = parse_bib(bib)
    r = _resolver(fetch)
    results = {k: verify_entry(e, r) for k, e in entries.items()}
    claims = []
    for c in citations(tex):
        res = results.get(c.key)
        if res is None:
            claims.append({"key": c.key, "line": c.line, "claim": c.claim, "label": "missing from .bib"})
            continue
        rec: Record | None = res.get("_record")
        src = (rec.abstract + " " + rec.fulltext) if rec else ""
        s = support(c.claim, re.sub(r"<[^>]+>", " ", src), nli)
        if s["label"] in ("unsupported", "partially supported", "contradicted") and "weak" not in res["badges"]:
            res["badges"].append("weak")
        claims.append({"key": c.key, "line": c.line, "claim": c.claim, **s})
    for v in results.values():
        v.pop("_record", None)
    return {"entries": results, "claims": claims}


def report_html(r: dict[str, Any]) -> str:
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{' '.join(BADGE[b] for b in v['badges'])}</td>"
                   f"<td>{html.escape('; '.join(v.get('mismatches', [])))}</td></tr>" for k, v in r["entries"].items())
    cl = "".join(f"<tr><td>{c['line']}</td><td>{html.escape(c['key'])}</td><td>{html.escape(c['claim'])}</td>"
                 f"<td>{c['label']}</td><td>{html.escape(' | '.join(e['sentence'] for e in c.get('evidence', [])))}</td></tr>"
                 for c in r["claims"])
    return ("<!doctype html><meta charset=utf-8><title>CiteGuard report</title><h1>CiteGuard report</h1>"
            f"<h2>References</h2><table border=1 cellpadding=4><tr><th>Key</th><th>Badges</th><th>Issues</th></tr>{rows}</table>"
            f"<h2>Claim support</h2><table border=1 cellpadding=4><tr><th>Line</th><th>Key</th><th>Claim</th><th>Label</th>"
            f"<th>Best source sentences</th></tr>{cl}</table>")


def fix_entry(e: BibEntry, rec: Record) -> str:
    """BibTeX for the canonical record, keeping the user's key."""
    authors = " and ".join(rec.authors)
    fields = {"title": f"{{{rec.title}}}", "author": f"{{{authors}}}", "year": f"{{{rec.year or ''}}}",
              "journal" if e.kind == "article" else "booktitle": f"{{{rec.venue}}}", "doi": f"{{{rec.doi or ''}}}"}
    return f"@{e.kind}{{{e.key},\n" + ",\n".join(f"  {k} = {v}" for k, v in fields.items()) + "\n}\n"


# ---------------- MCP tools ----------------

def verify_bib(bib_path: str) -> dict[str, Any]:
    """Check every .bib entry against Crossref/OpenAlex: existence, metadata mismatch, retraction, newer version."""
    r = _resolver()
    out = {k: verify_entry(e, r) for k, e in parse_bib(Path(roots.check(bib_path)).read_text(encoding="utf-8")).items()}
    for v in out.values():
        v.pop("_record", None)
    return out


def check_claims(tex_path: str, bib_path: str) -> list[dict[str, Any]]:
    """Does each cited work support the sentence it is cited for? (NLI runs locally.)"""
    return check(Path(roots.check(tex_path)).read_text(encoding="utf-8"),
                 Path(roots.check(bib_path)).read_text(encoding="utf-8"), nli=get_nli("auto"))["claims"]


def check_claim(claim: str, source_text: str) -> dict[str, Any]:
    """Does this source text (e.g. an abstract) support, contradict, or not address the claim?"""
    return support(claim, source_text, get_nli("auto"))


def fix_bib_entry(bib_path: str, key: str) -> str:
    """Canonical BibTeX for one entry (returned as text; the .bib file is not modified)."""
    e = parse_bib(Path(roots.check(bib_path)).read_text(encoding="utf-8"))[key]
    rec, _ = resolve(e, _resolver())
    if rec is None:
        raise LookupError(f"no canonical record found for {key}")
    return fix_entry(e, rec)


TOOLS = [verify_bib, check_claims, check_claim, fix_bib_entry]
