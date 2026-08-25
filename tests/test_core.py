from recap import Candidate, RecapEngine, SIDTrie


def test_legal_normalization_and_fixed_budget_reintegration() -> None:
    trie = SIDTrie([("a", "x"), ("a", "y"), ("b", "z")])

    def locator(_evidence):
        return 1, 0.95

    def repairer(prefix, _evidence, _trie):
        assert prefix == ("a",)
        return ("a", "y")

    engine = RecapEngine(
        trie, locator, repairer,
        error_threshold=0.8, action_threshold=0.1, max_reciprocal_entropy=1.0,
    )
    anchor = Candidate(("a", "x"), "item-x", -1.0)
    evidence = engine.evidence_for(
        anchor.sid,
        lambda prefix: {"a": 1.0, "b": 0.0} if not prefix else {"x": 0.1, "y": 0.2},
        lambda prefix: {"a": 0.5, "b": 0.2} if not prefix else {"x": 0.0, "y": 1.0},
    )
    assert set(evidence[1].direct) == {"x", "y"}
    assert round(sum(evidence[1].reciprocal.values()), 8) == 1.0
    proposal, decision = engine.revise(anchor, evidence, lambda sid: -0.4 if sid[-1] == "y" else -1.0, lambda sid: f"item-{sid[-1]}")
    assert decision.accepted
    assert proposal.sid == ("a", "y")
    beam = engine.reintegrate([anchor, Candidate(("b", "z"), "item-z", -0.6)], proposal, width=2)
    assert [candidate.item_id for candidate in beam] == ["item-y", "item-z"]


def test_non_actionable_gate_keeps_anchor() -> None:
    trie = SIDTrie([("a", "x"), ("a", "y")])
    engine = RecapEngine(
        trie, lambda _evidence: (1, 0.99), lambda prefix, _evidence, _trie: (*prefix, "y"),
        error_threshold=0.8, action_threshold=5.0, max_reciprocal_entropy=1.0,
    )
    anchor = Candidate(("a", "x"), "item-x", -1.0)
    evidence = engine.evidence_for(
        anchor.sid,
        lambda prefix: {"a": 0.0} if not prefix else {"x": 0.0, "y": 0.0},
        lambda prefix: {"a": 0.0} if not prefix else {"x": 0.0, "y": 0.1},
    )
    proposal, decision = engine.revise(anchor, evidence, lambda _sid: -0.2, lambda sid: sid)
    assert not decision.accepted
    assert decision.reason == "non-actionable"
    assert proposal == anchor
