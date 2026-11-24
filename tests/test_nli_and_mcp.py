"""NLI wrapper (with a stand-in cross-encoder: no download), classification and calibration
branches, the `claim` command, and the MCP audit / read-only-roots layer."""
import sqlite3
import sys
import types

import numpy as np
import pytest

from citeguard import api
from citeguard.algorithms import support as sp
from citeguard.cli import main

ABSTRACT = ("We propose residual routing for wireless mesh networks. Residual routing reduces packet latency "
            "in mesh networks by learning shortcut paths between distant nodes. Experiments on three testbeds "
            "confirm lower latency. We also release our simulator for reproducibility.")


class StubCE:
    """Mimics sentence_transformers.CrossEncoder: label order deliberately differs from the usual."""

    def __init__(self, model, device=None):
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(
            id2label={0: "NEUTRAL", 1: "Contradiction", 2: "ENTAILMENT"}))

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        out = []
        for premise, hyp in pairs:
            ent = 5.0 if hyp.lower().rstrip(".") in premise.lower() else -2.0
            con = 5.0 if "not" in hyp.lower() and hyp.lower().replace("not ", "").rstrip(".") in premise.lower() else -2.0
            out.append([0.0, con, ent])
        return np.array(out)


@pytest.fixture
def stub_st(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=StubCE))
    api._NLI_CACHE.clear()
    yield
    api._NLI_CACHE.clear()


def test_cross_encoder_wrapper_reads_label_map(stub_st):
    nli = sp.CrossEncoderNLI()
    pe, pc = nli("residual routing reduces packet latency in mesh networks", "Residual routing reduces packet latency")
    assert pe > 0.9 and pc < 0.05
    assert nli.batch([]) == []
    s = sp.support("Residual routing reduces packet latency in mesh networks by learning shortcut paths between distant nodes",
                   ABSTRACT, nli)
    assert s["label"] == "supported"
    c = sp.support("Residual routing does not reduce packet latency in mesh networks by learning shortcut paths between distant nodes",
                   ABSTRACT, nli)
    assert c["label"] in ("contradicted", "unsupported")


def test_support_accepts_presplit_sentences_and_abstains():
    sents = ["Residual routing reduces packet latency in mesh networks.", "Three testbeds were used here.",
             "The simulator is released openly online."]
    r = sp.support("Residual routing reduces packet latency in mesh networks", sents)
    assert r["label"] == "supported" and r["evidence"][0]["offset"] == 0
    assert sp.support("x", ["too short", "also short"])["label"] == "cannot verify"


def test_classify_branches_and_3way_calibration():
    th = sp.Thresholds(0.6, 0.3, 0.5)
    assert sp.classify(0.7, 0.1, th) == "supported"
    assert sp.classify(0.7, 0.9, th) == "contradicted"
    assert sp.classify(0.4, 0.1, th) == "partially supported"
    assert sp.classify(0.1, 0.1, th) == "unsupported"
    sup = [0.9, 0.8, -0.5, -0.6, 0.0, 0.1]
    con = [-0.9, -0.8, 0.7, 0.8, 0.0, 0.1]
    labels = ["SUPPORT", "SUPPORT", "CONTRADICT", "CONTRADICT", "NEI", "NEI"]
    t, f1 = sp.calibrate_3way(sup, con, labels, grid=21)
    assert f1 == pytest.approx(1.0) and t.partial == pytest.approx(t.supported / 2)


def test_get_nli_backends(stub_st, monkeypatch):
    assert api.get_nli("lexical") is sp.lexical_nli
    assert isinstance(api.get_nli("cross-encoder"), sp.CrossEncoderNLI)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # simulate "not installed"
    api._NLI_CACHE.clear()
    assert api.get_nli("auto") is sp.lexical_nli
    api._NLI_CACHE.clear()
    with pytest.raises(SystemExit, match=r"citeguard\[nli\]"):
        api.get_nli("cross-encoder")


def test_claim_command_and_tool(tmp_path, capsys):
    (tmp_path / "abs.txt").write_text(ABSTRACT)
    assert main(["claim", "Residual routing reduces packet latency in mesh networks", str(tmp_path / "abs.txt"),
                 "--nli", "lexical"]) == 0
    assert '"label": "supported"' in capsys.readouterr().out


def test_mcp_layer_audit_and_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("CITEGUARD_HOME", str(tmp_path))
    from citeguard.mcp_server import audited
    f = tmp_path / "refs.bib"
    f.write_text("@article{a, title={T}, year={2020}}")
    calls = []
    audited(lambda path: calls.append(path) or "ok")(path=str(f))
    with pytest.raises(ZeroDivisionError):
        audited(lambda: 1 / 0)()
    rows = sqlite3.connect(tmp_path / "audit.db").execute("select ok from audit").fetchall()
    assert rows == [(1,), (0,)]
    from citeguard import roots
    monkeypatch.setenv("CITEGUARD_ROOTS", str(tmp_path / "allowed"))
    with pytest.raises(PermissionError):
        roots.check(f)
    monkeypatch.setenv("CITEGUARD_ROOTS", str(tmp_path))
    assert roots.check(f) == f.resolve()


def test_mcp_server_registers_tools():
    pytest.importorskip("mcp")
    import asyncio

    from citeguard.mcp_server import build_server
    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert names == {"verify_bib", "check_claims", "check_claim", "fix_bib_entry"}
