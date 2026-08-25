"""RECAP core: constrained reciprocal critique and SID suffix repair."""

from .core import Candidate, EvidenceFeatures, GateDecision, PoERepairer, RecapEngine, RefinementResult, RouteEvidence
from .rollout import RolloutInput, StaticLogitTable, trie_from_json
from .trie import SIDTrie

__all__ = [
    "Candidate", "EvidenceFeatures", "GateDecision", "PoERepairer",
    "RecapEngine", "RefinementResult", "RolloutInput", "RouteEvidence",
    "SIDTrie", "StaticLogitTable", "trie_from_json",
]
