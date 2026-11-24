"""Metadata clients (Crossref, OpenAlex, arXiv) with an SQLite TTL cache and polite rate limits.

Privacy: only titles and DOIs leave the machine; manuscript text never does.
Set CITEGUARD_MAILTO to identify yourself to Crossref/OpenAlex polite pools (required by their
etiquette); it is never set automatically.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from collections.abc import Callable

from ..algorithms.matching import Record

Fetch = Callable[[str], Any]


class Cache:
    def __init__(self, path: Path | None = None, ttl: float = 30 * 86400) -> None:
        p = path or Path(os.environ.get("CITEGUARD_HOME", Path.home() / ".citeguard")) / "cache.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(p, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("CREATE TABLE IF NOT EXISTS c(url TEXT PRIMARY KEY, ts REAL, body TEXT)")
        self.ttl = ttl

    def get(self, url: str) -> Any | None:
        r = self.con.execute("SELECT ts, body FROM c WHERE url=?", (url,)).fetchone()
        return json.loads(r[1]) if r and time.time() - r[0] < self.ttl else None

    def put(self, url: str, body: Any) -> None:
        self.con.execute("INSERT OR REPLACE INTO c VALUES (?,?,?)", (url, time.time(), json.dumps(body)))
        self.con.commit()


class Polite:
    """Wraps a fetch function with caching and a minimum interval per host."""

    def __init__(self, fetch: Fetch | None = None, cache: Cache | None = None, min_interval: float = 0.2) -> None:
        self.fetch = fetch or self._http
        self.cache = cache
        self.min_interval = min_interval
        self.last: dict[str, float] = {}
        self.lock = threading.Lock()
        self.calls = 0

    @staticmethod
    def _http(url: str) -> Any:  # pragma: no cover - network
        mailto = os.environ.get("CITEGUARD_MAILTO")
        ua = "citeguard/0.1" + (f" (mailto:{mailto})" if mailto else "")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}), timeout=30) as r:
            data = r.read()
        return json.loads(data) if data[:1] in (b"{", b"[") else data.decode("utf-8", "replace")

    def __call__(self, url: str) -> Any:
        if self.cache and (hit := self.cache.get(url)) is not None:
            return hit
        host = urllib.parse.urlparse(url).hostname or ""
        with self.lock:
            wait = self.last.get(host, 0) + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.time()
        body = self.fetch(url)
        self.calls += 1
        if self.cache:
            self.cache.put(url, body)
        return body


def _crossref_record(m: dict[str, Any]) -> Record:
    year = None
    for k in ("published-print", "published-online", "issued", "created"):
        dp = m.get(k, {}).get("date-parts")
        if dp and dp[0] and dp[0][0]:
            year = int(dp[0][0])
            break
    rel = {k: [x.get("id", "") for x in v] for k, v in m.get("relation", {}).items()}
    return Record(title=(m.get("title") or [""])[0],
                  authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
                  year=year, venue=(m.get("container-title") or [""])[0], doi=m.get("DOI"), source="crossref",
                  abstract=m.get("abstract", ""), is_preprint=m.get("subtype") == "preprint" or m.get("type") == "posted-content",
                  relations=rel, updates=[{"type": u.get("type", ""), "DOI": u.get("DOI", "")}
                                          for u in m.get("updated-by", []) + m.get("update-to", [])])


def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


class Resolver:
    def __init__(self, fetch: Fetch) -> None:
        self.fetch = fetch

    def by_doi(self, doi: str) -> Record | None:
        try:
            m = self.fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")["message"]
        except Exception:  # noqa: BLE001
            return None
        rec = _crossref_record(m)
        self._enrich_openalex(rec)
        return rec

    def by_title(self, title: str, rows: int = 5) -> list[Record]:
        q = urllib.parse.quote(title)
        try:
            items = self.fetch(f"https://api.crossref.org/works?query.bibliographic={q}&rows={rows}")["message"]["items"]
        except Exception:  # noqa: BLE001
            return []
        return [_crossref_record(m) for m in items]

    def _enrich_openalex(self, rec: Record) -> None:
        if not rec.doi:
            return
        try:
            w = self.fetch(f"https://api.openalex.org/works/doi:{urllib.parse.quote(rec.doi)}")
        except Exception:  # noqa: BLE001
            return
        rec.abstract = rec.abstract or _openalex_abstract(w.get("abstract_inverted_index"))
        if rec.is_preprint:
            for loc in w.get("locations", []):
                d = (loc.get("landing_page_url") or "")
                if "doi.org/" in d and rec.doi.lower() not in d.lower():
                    pub = self.by_doi(d.split("doi.org/")[1])
                    if pub and not pub.is_preprint:
                        rec.related_published.append(pub)
