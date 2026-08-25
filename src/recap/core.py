"""Paper-aligned RECAP inference primitives.

This module intentionally owns no backbone. A caller provides route logits on
the candidate prefix and the frozen backbone's direct score. This keeps RECAP
from silently changing a backbone, fusion function, or beam budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Callable, Hashable, Protocol, Sequence

from .trie import SIDTrie

Token = Hashable


def _softmax_on_legal(logits: dict[Token, float], legal: Sequence[Token]) -> dict[Token, float]:
    if not legal:
        return {}
    missing = [token for token in legal if token not in logits]
    if missing:
        raise KeyError(f"Route logits omit legal tokens: {missing!r}")
    offset = max(logits[token] for token in legal)
    weights = {token: exp(logits[token] - offset) for token in legal}
    total = sum(weights.values())
    return {token: weight / total for token, weight in weights.items()}


def _entropy(probabilities: dict[Token, float]) -> float:
    return -sum(p * log(p) for p in probabilities.values() if p > 0.0)


@dataclass(frozen=True)
class Candidate:
    sid: tuple[Token, ...]
    item_id: Hashable
    direct_score: float


@dataclass(frozen=True)
class RouteEvidence:
    """Legal-set-normalized beliefs for one committed SID token."""

    prefix: tuple[Token, ...]
    committed: Token
    direct: dict[Token, float]
    reciprocal: dict[Token, float]

    @property
    def reciprocal_entropy(self) -> float:
        return _entropy(self.reciprocal)

    def actionability_margin(self, mixing_weight: float) -> float:
        """Best legal alternative minus the committed token under log-PoE."""
        if not 0.0 <= mixing_weight <= 1.0:
            raise ValueError("mixing_weight must lie in [0, 1].")
        alternatives = [token for token in self.direct if token != self.committed]
        if not alternatives:
            return float("-inf")

        def score(token: Token) -> float:
            return mixing_weight * log(self.direct[token]) + (1.0 - mixing_weight) * log(self.reciprocal[token])

        return max(score(token) for token in alternatives) - score(self.committed)


@dataclass(frozen=True)
class GateDecision:
    level: int | None
    locator_confidence: float
    actionability_margin: float
    reciprocal_entropy: float
    reason: str

    @property
    def accepted(self) -> bool:
        return self.level is not None


class Locator(Protocol):
    def __call__(self, evidence: Sequence[RouteEvidence]) -> tuple[int | None, float]:
        """Return zero-based first-error level (or None) and confidence."""


class Repairer(Protocol):
    def __call__(self, prefix: tuple[Token, ...], evidence: Sequence[RouteEvidence], trie: SIDTrie) -> tuple[Token, ...]:
        """Return a full legal SID beginning with ``prefix``."""


class RecapEngine:
    """Selectively repair one anchor and reintegrate it without increasing B."""

    def __init__(
        self,
        trie: SIDTrie,
        locator: Locator,
        repairer: Repairer,
        *,
        error_threshold: float,
        action_threshold: float,
        max_reciprocal_entropy: float,
        mixing_weight: float = 0.5,
    ) -> None:
        self.trie = trie
        self.locator = locator
        self.repairer = repairer
        self.error_threshold = error_threshold
        self.action_threshold = action_threshold
        self.max_reciprocal_entropy = max_reciprocal_entropy
        self.mixing_weight = mixing_weight

    def evidence_for(
        self,
        sid: Sequence[Token],
        direct_logits: Callable[[tuple[Token, ...]], dict[Token, float]],
        reciprocal_logits: Callable[[tuple[Token, ...]], dict[Token, float]],
    ) -> list[RouteEvidence]:
        """Evaluate both routes under the *same realized direct prefix*."""
        if not self.trie.is_legal(sid):
            raise ValueError("The audited anchor must be a terminal legal SID.")
        evidence: list[RouteEvidence] = []
        prefix: tuple[Token, ...] = ()
        for token in sid:
            legal = self.trie.legal_children(prefix)
            direct = _softmax_on_legal(direct_logits(prefix), legal)
            reciprocal = _softmax_on_legal(reciprocal_logits(prefix), legal)
            evidence.append(RouteEvidence(prefix, token, direct, reciprocal))
            prefix = (*prefix, token)
        return evidence

    def decide(self, evidence: Sequence[RouteEvidence]) -> GateDecision:
        level, confidence = self.locator(evidence)
        if level is None:
            return GateDecision(None, confidence, float("nan"), float("nan"), "no-error")
        if not 0 <= level < len(evidence):
            raise ValueError("Locator returned an out-of-range level.")
        step = evidence[level]
        margin = step.actionability_margin(self.mixing_weight)
        entropy = step.reciprocal_entropy
        if confidence < self.error_threshold:
            return GateDecision(None, confidence, margin, entropy, "low-locator-confidence")
        if margin < self.action_threshold:
            return GateDecision(None, confidence, margin, entropy, "non-actionable")
        if entropy > self.max_reciprocal_entropy:
            return GateDecision(None, confidence, margin, entropy, "diffuse-reciprocal-evidence")
        return GateDecision(level, confidence, margin, entropy, "accepted")

    def revise(
        self,
        anchor: Candidate,
        evidence: Sequence[RouteEvidence],
        direct_score: Callable[[tuple[Token, ...]], float],
        resolve_item: Callable[[tuple[Token, ...]], Hashable],
    ) -> tuple[Candidate, GateDecision]:
        decision = self.decide(evidence)
        if not decision.accepted:
            return anchor, decision
        assert decision.level is not None
        proposal = self.repairer(tuple(anchor.sid[: decision.level]), evidence[decision.level :], self.trie)
        if not proposal[: decision.level] == anchor.sid[: decision.level] or not self.trie.is_legal(proposal):
            return anchor, GateDecision(None, decision.locator_confidence, decision.actionability_margin, decision.reciprocal_entropy, "illegal-repair")
        return Candidate(proposal, resolve_item(proposal), direct_score(proposal)), decision

    @staticmethod
    def reintegrate(beam: Sequence[Candidate], proposal: Candidate, width: int) -> list[Candidate]:
        """Deduplicate by item, retain only top direct scores, and keep width fixed."""
        if width <= 0:
            raise ValueError("width must be positive.")
        by_item: dict[Hashable, Candidate] = {}
        for candidate in (*beam, proposal):
            previous = by_item.get(candidate.item_id)
            if previous is None or candidate.direct_score > previous.direct_score:
                by_item[candidate.item_id] = candidate
        return sorted(by_item.values(), key=lambda candidate: candidate.direct_score, reverse=True)[:width]
