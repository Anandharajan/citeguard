"""All network access is replaced by recorded-style fixtures (no network in unit tests)."""
import numpy as np
import pytest

from citeguard import api
from citeguard.algorithms.drift import drift_flags, published_version
from citeguard.algorithms.matching import Calibrator, Record, mismatches, raw_score
from citeguard.algorithms.support import calibrate_thresholds, lexical_nli, split_sentences, support
from citeguard.cli import main
from citeguard.io.clients import Cache, Polite, Resolver
from citeguard.io.texbib import citations, parse_bib

BIB = r"""
@article{smith2020,
  title = {Deep {Residual} Routing for Mesh Networks},
  author = {Smith, John and Lee, Ann},
  journal = {IEEE Trans. Networking},
  year = {2020},
  doi = {10.1000/route.1}
}
@inproceedings{fake2023,
  title = "Quantum Gravity Improves Spreadsheet Accuracy",
  author = {Nobody, A.},
  year = 2023
}
@article{pre2021, title={Sparse Attention for Channel Estimation}, author={Kumar, R. and Das, P.}, year={2021}, doi={10.48550/arxiv.2101.00001}}
@article{retr2019, title={Miracle Battery Chemistry}, author={Doe, Jane}, year={2019}, doi={10.1000/retr.9}}
"""

TEX = r"""\section{Intro}
Residual routing reduces packet latency in mesh networks by learning shortcut paths \cite{smith2020}.
% commented \cite{nothere}
Spreadsheets are improved \citep[p.~3]{fake2023}. Sparse attention fails to estimate channels accurately \cite{pre2021}.
Battery capacity doubled \cite{retr2019,missing2024}.
"""

ABSTRACT = ("We propose residual routing for wireless mesh networks. Residual routing reduces packet latency "
            "in mesh networks by learning shortcut paths between distant nodes. Experiments on three testbeds "
            "confirm lower latency. We also release our simulator for reproducibility.")


def cr(doi, title, authors, year, **extra):
    fam = [{"given": a.split()[0], "family": a.split()[-1]} for a in authors]
    return {"message": {"DOI": doi, "title": [title], "author": fam, "issued": {"date-parts": [[year]]},
                        "container-title": ["Venue"], **extra}}


FIX = {
    "https://api.crossref.org/works/10.1000/route.1": cr("10.1000/route.1", "Deep Residual Routing for Mesh Networks",
                                                         ["John Smith", "Ann Lee", "Bo Chen"], 2020),
    "https://api.openalex.org/works/doi:10.1000/route.1": {"abstract_inverted_index": None, "locations": []},
    "https://api.crossref.org/works/10.48550/arxiv.2101.00001": cr(
        "10.48550/arxiv.2101.00001", "Sparse Attention for Channel Estimation", ["R Kumar", "P Das"], 2021,
        type="posted-content", subtype="preprint",
        abstract="Sparse attention improves channel estimation accuracy. We test on 5G traces. It is fast and simple to deploy here."),
    "https://api.openalex.org/works/doi:10.48550/arxiv.2101.00001": {"locations": [
        {"landing_page_url": "https://doi.org/10.1000/pub.7"}]},
    "https://api.crossref.org/works/10.1000/pub.7": cr("10.1000/pub.7", "Sparse Attention for Channel Estimation",
                                                       ["R Kumar", "P Das"], 2022),
    "https://api.openalex.org/works/doi:10.1000/pub.7": {"locations": []},
    "https://api.crossref.org/works/10.1000/retr.9": cr("10.1000/retr.9", "Miracle Battery Chemistry", ["Jane Doe"], 2019,
                                                        **{"updated-by": [{"type": "retraction", "DOI": "10.1000/notice.1"}]}),
    "https://api.openalex.org/works/doi:10.1000/retr.9": {"locations": []},
}
FIX["https://api.crossref.org/works/10.1000/route.1"]["message"]["abstract"] = ABSTRACT


def fetch(url):
    if url in FIX:
        return FIX[url]
    if "query.bibliographic" in url:
        return {"message": {"items": [cr("10.9/x", "An Unrelated Paper About Soil", ["Q Zed"], 2010)["message"]]}}
    raise OSError("404")


def test_parse_bib_and_cites():
    b = parse_bib(BIB)
    assert set(b) == {"smith2020", "fake2023", "pre2021", "retr2019"}
    assert b["smith2020"].title == "Deep Residual Routing for Mesh Networks" and b["fake2023"].year == 2023
    assert b["smith2020"].authors == ["Smith, John", "Lee, Ann"] and b["pre2021"].doi.startswith("10.48550")
    cs = citations(TEX)
    keys = [c.key for c in cs]
    assert keys == ["smith2020", "fake2023", "pre2021", "retr2019", "missing2024"]
    assert "\\cite" not in cs[0].claim and cs[0].line == 2
    assert cs[1].claim.startswith("Residual routing")  # short claim gets the previous sentence as context


def test_matching_and_calibration():
    rec = Record("Deep Residual Routing for Mesh Networks", ["John Smith", "Ann Lee"], 2020, "IEEE Trans. Networking")
    good = raw_score("Deep Residual Routing for Mesh Networks", ["Smith, John", "Lee, Ann"], 2020, "IEEE Trans. Networking", rec)
    bad = raw_score("Quantum Gravity Improves Spreadsheets", ["Nobody, A."], 2023, "", rec)
    assert good["score"] > 0.95 > 0.5 > bad["score"]
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.uniform(0.7, 1, 100), rng.uniform(0, 0.6, 100)])
    y = np.r_[np.ones(100), np.zeros(100)]
    cal = Calibrator().fit(s, y)
    assert cal(0.95) > 0.9 and cal(0.2) < 0.1
    assert mismatches("X", ["A B"], 2020, Record("Y", ["A B", "C D"], 2021))[0].startswith("title")


def test_support_scoring():
    assert support("Residual routing reduces latency.", "Too short. Two.")["label"] == "cannot verify"
    s = support("Residual routing reduces packet latency in mesh networks", ABSTRACT)
    assert s["label"] == "supported" and s["evidence"][0]["sentence"].startswith("Residual routing reduces")
    assert ABSTRACT[s["evidence"][0]["offset"]:].startswith("Residual routing")
    assert support("Residual routing does not reduce packet latency in mesh networks", ABSTRACT)["label"] == "contradicted"
    assert support("Quantum gravity improves spreadsheet accuracy", ABSTRACT)["label"] == "unsupported"
    assert lexical_nli("", "") == (0.0, 0.0) and len(split_sentences(ABSTRACT)) == 4
    th = calibrate_thresholds([0.9, 0.8, 0.5, 0.4, 0.1, 0.0], ["supported"] * 2 + ["partially supported"] * 2 + ["unsupported"] * 2)
    assert 0.1 < th.partial <= 0.4 and 0.5 < th.supported <= 0.8


def test_drift():
    pre = Record("Sparse Attention for Channel Estimation", ["R Kumar", "P Das"], 2021, is_preprint=True)
    pub = Record("Sparse attention for channel estimation", ["R. Kumar", "P. Das"], 2022, doi="10.1/p")
    other = Record("Sparse Attention for Channel Estimation", ["X Y"], 2022, doi="10.1/o")
    assert published_version(pre, [other, pub]) is pub
    assert published_version(Record("T", [], 2020, is_preprint=True, relations={"is-preprint-of": ["10.1/P"]}), [pub]) is pub
    assert drift_flags(Record("t", [], 2020, updates=[{"type": "expression-of-concern"}, {"type": "erratum"}])) == [
        {"flag": "expression of concern", "notice": ""}, {"flag": "corrected", "notice": ""}]


def test_end_to_end_badges(tmp_path, capsys):
    r = api.check(TEX, BIB, fetch=fetch)
    e = r["entries"]
    assert e["smith2020"]["badges"][0] == "mismatch" and "chen" in e["smith2020"]["mismatches"][0]
    assert e["fake2023"]["badges"] == ["not_found"]
    assert "newer" in e["pre2021"]["badges"]
    assert "retracted" in e["retr2019"]["badges"]
    labels = {c["key"]: c["label"] for c in r["claims"]}
    assert labels["smith2020"] == "supported" and labels["missing2024"] == "missing from .bib"
    assert labels["pre2021"] == "unsupported"  # claim contradicts the abstract ("fails")
    assert labels["fake2023"] == "cannot verify"
    (tmp_path / "m.tex").write_text(TEX)
    (tmp_path / "r.bib").write_text(BIB)
    assert main(["check", str(tmp_path / "m.tex"), str(tmp_path / "r.bib"), "--report", str(tmp_path / "r.html"), "--nli", "lexical"], fetch=fetch) == 0
    assert "CiteGuard report" in (tmp_path / "r.html").read_text(encoding="utf-8")
    assert main(["check", str(tmp_path / "m.tex"), str(tmp_path / "r.bib"), "--json", "--nli", "lexical"], fetch=fetch) == 0
    fixed = api.fix_entry(parse_bib(BIB)["smith2020"], Resolver(fetch).by_doi("10.1000/route.1"))
    assert "Bo Chen" in fixed and "@article{smith2020" in fixed


def test_cache_and_rate_limit(tmp_path):
    calls = []
    p = Polite(fetch=lambda u: calls.append(u) or {"ok": 1}, cache=Cache(tmp_path / "c.db"), min_interval=0.05)
    import time
    t = time.time()
    assert p("https://api.crossref.org/a") == {"ok": 1} and p("https://api.crossref.org/a") == {"ok": 1}
    p("https://api.crossref.org/b")
    assert calls == ["https://api.crossref.org/a", "https://api.crossref.org/b"]  # second call served from cache
    assert time.time() - t >= 0.05
    expired = Cache(tmp_path / "c.db", ttl=0)
    assert expired.get("https://api.crossref.org/a") is None


def test_speed_200_entries_warm_cache(tmp_path):
    bib = "\n".join(f"@article{{k{i}, title={{Deep Residual Routing for Mesh Networks}}, author={{Smith, John and Lee, Ann}}, "
                    f"year={{2020}}, doi={{10.1000/route.1}}}}" for i in range(200))
    tex = " ".join(f"Residual routing reduces packet latency in mesh networks \\cite{{k{i}}}." for i in range(200))
    p = Polite(fetch=fetch, cache=Cache(tmp_path / "c.db"), min_interval=0)
    api.check(tex, bib, fetch=p)  # warm
    import time
    t = time.time()
    api.check(tex, bib, fetch=p)
    assert time.time() - t < 60


def test_resolver_failures():
    r = Resolver(lambda u: (_ for _ in ()).throw(OSError()))
    assert r.by_doi("10.1/x") is None and r.by_title("x") == []


def test_fix_bib_entry_not_found(tmp_path, monkeypatch):
    (tmp_path / "r.bib").write_text(BIB)
    monkeypatch.setattr(api, "_resolver", lambda fetch_=None: Resolver(fetch))
    with pytest.raises(LookupError):
        api.fix_bib_entry(str(tmp_path / "r.bib"), "fake2023")
    assert "Bo Chen" in api.fix_bib_entry(str(tmp_path / "r.bib"), "smith2020")
