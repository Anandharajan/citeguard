"""Minimal BibTeX parser and \\cite extractor (citing sentence + previous sentence as context)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class BibEntry:
    key: str
    kind: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return re.sub(r"[{}]", "", self.fields.get("title", "")).strip()

    @property
    def authors(self) -> list[str]:
        a = self.fields.get("author", "")
        return [x.strip() for x in re.split(r"\s+and\s+", a) if x.strip()]

    @property
    def year(self) -> int | None:
        m = re.search(r"\d{4}", self.fields.get("year", ""))
        return int(m.group()) if m else None

    @property
    def doi(self) -> str | None:
        d = self.fields.get("doi", "").strip()
        return re.sub(r"^https?://(dx\.)?doi\.org/", "", d) or None

    @property
    def venue(self) -> str:
        return self.fields.get("journal") or self.fields.get("booktitle") or ""


def _value(s: str, i: int) -> tuple[str, int]:
    while s[i].isspace():
        i += 1
    if s[i] == "{":
        depth, j = 0, i
        while True:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
    if s[i] == '"':
        j = s.index('"', i + 1)
        return s[i + 1:j], j + 1
    m = re.match(r"[^,}\s]+", s[i:])
    assert m
    return m.group(), i + m.end()


def parse_bib(text: str) -> dict[str, BibEntry]:
    out: dict[str, BibEntry] = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = m.group(1).lower(), m.group(2)
        if kind in ("comment", "preamble", "string"):
            continue
        i, fields = m.end(), {}
        while i < len(text):
            fm = re.match(r"\s*(\w[\w-]*)\s*=\s*", text[i:])
            if not fm:
                break
            name = fm.group(1).lower()
            val, i = _value(text, i + fm.end())
            fields[name] = re.sub(r"\s+", " ", val).strip()
            cm = re.match(r"\s*,?", text[i:])
            i += cm.end() if cm else 0
            if re.match(r"\s*\}", text[i:]):
                break
        out[key] = BibEntry(key, kind, fields)
    return out


@dataclass
class Citation:
    key: str
    claim: str
    line: int


_CITE = re.compile(r"\\(?:cite|citep|citet|parencite|textcite|autocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}")


_MASK = re.compile(r"⟦CITE:([^⟧]+)⟧")


def citations(tex: str) -> list[Citation]:
    body = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in tex.splitlines())
    # mask cite commands first: optional args like [p.~3] contain dots that would split sentences
    body = _CITE.sub(lambda m: "⟦CITE:" + m.group(1) + "⟧", body)
    # Strip structural markup without shifting original line numbers.
    def blank(m: re.Match[str]) -> str:
        return "".join("\n" if ch == "\n" else " " for ch in m.group())

    body = re.sub(r"\\(?:begin|end)\{[^}]+\}(?:\{[^{}]*\})?", blank, body)
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline|cline(?:\{[^}]*\})?)\b", blank, body)
    body = re.sub(r"\\\\(?:\[[^\]]*\])?", lambda m: "." + " " * (len(m.group()) - 1), body)
    body = re.sub(r"(?<!\\)&", " ", body)
    body = re.sub(r"\\(?:eqref|ref|pageref|label)\{[^}]*\}", blank, body)
    body = re.sub(r"\\(?:documentclass|usepackage)(?:\[[^\]]*\])?\{[^}]*\}", blank, body)
    out = []
    prev = ""
    for sm in re.finditer(r"[\s\S]+?(?:[.!?]+(?=\s|$)|$)", body):
        s = sm.group()
        if not s.strip():
            continue
        claim = _MASK.sub("", s)
        claim = re.sub(r"\\(?:(?:sub)*section|chapter|paragraph|label|caption)\*?\{[^}]*\}", " ", claim)
        claim = re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s+", " ", claim)).strip()
        for cm in _MASK.finditer(s):
            line = body.count("\n", 0, sm.start() + cm.start()) + 1
            for k in cm.group(1).split(","):
                out.append(Citation(k.strip(), (prev + " " + claim).strip() if len(claim.split()) < 6 else claim, line))
        prev = claim
    return out
