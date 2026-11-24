"""citeguard check main.tex refs.bib --report report.html"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import api


def main(argv: list[str] | None = None, fetch=None) -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(prog="citeguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("tex", type=Path)
    c.add_argument("bib", type=Path)
    c.add_argument("--report", type=Path)
    c.add_argument("--json", action="store_true")
    c.add_argument("--nli", choices=["auto", "cross-encoder", "lexical"], default="auto",
                   help="claim-support scorer (default: local DeBERTa NLI if installed)")
    q = sub.add_parser("claim", help="check one claim against a source text (no network)")
    q.add_argument("claim")
    q.add_argument("source", type=Path, help="text file with the abstract or passage")
    q.add_argument("--nli", choices=["auto", "cross-encoder", "lexical"], default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "claim":
        r1 = api.support(a.claim, a.source.read_text(encoding="utf-8"), api.get_nli(a.nli))
        print(json.dumps(r1, indent=2))
        return 0
    r = api.check(a.tex.read_text(encoding="utf-8"), a.bib.read_text(encoding="utf-8"), fetch=fetch, nli=api.get_nli(a.nli))
    if a.json:
        print(json.dumps(r, indent=2, default=str))
    else:
        for k, v in r["entries"].items():
            print(f"{' '.join(api.BADGE[b] for b in v['badges']):6} {k}  {'; '.join(v.get('mismatches', []))}")
        for cl in r["claims"]:
            if cl["label"] != "supported":
                print(f"  line {cl['line']} [{cl['key']}] {cl['label']}: {cl['claim'][:80]}")
    if a.report:
        a.report.write_text(api.report_html(r), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
