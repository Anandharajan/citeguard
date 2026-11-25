from citeguard.algorithms.trace import assess, atomic_claims, evidence_graph, quantitative, quantity


def test_numeric_units_and_mismatch():
    assert quantitative("Latency decreased by 1 s.", "Latency decreased by 1000 ms.")["status"] == "match"
    assert quantitative("Latency decreased by 12%.", "Latency decreased by 42%.")["status"] == "potential mismatch"
    assert quantitative("Latency decreased by 12%.", "Energy decreased by 42%.")["status"] == "not comparable"
    assert quantitative("Latency decreased by 12% at site A.", "Latency decreased by 42% at site B.")["status"] == "not comparable"


def test_escaped_percent_and_qualifier():
    assert quantity(r"Latency decreased by 12\%.")["value"] == 12
    assert quantity("Latency decreased by approximately 12%.") is None
    assert quantity("Latency was not decreased by 12%.") is None


def test_atomic_and_group():
    claim = "The system was highly accurate; the method was computationally efficient."
    assert len(atomic_claims(claim)) == 2
    rows = [{"line": 1, "claim": claim, "key": k} for k in ("a", "b")]
    graph = evidence_graph(rows, {})
    assert len(graph[0]["edges"]) == 4
    assert graph[0]["coverage"] == 0


def test_quantitative_scan_and_fulltext():
    claim = "Latency decreased by 12%."
    source = "Latency decreased by 42%. We used three datasets. All experiments were repeated."
    result = assess(claim, source)
    assert result["label"] == "requires review"
    assert result["quantitative"][0]["status"] == "potential mismatch"
    full = "Latency decreased by 12%. We used three datasets. All experiments were repeated."
    result = assess(claim, "", fulltext=full)
    assert result["route"] == "provided full text"
    assert result["quantitative"][0]["status"] == "match"
    assert not result["calibrated"]
