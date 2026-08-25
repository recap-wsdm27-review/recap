"""Portable, review-safe inputs for exercising the RECAP inference core.

This module deliberately stores only an already-produced native beam and
prefix-indexed route logits. It neither loads a benchmark nor reconstructs a
backbone; those responsibilities remain with a versioned adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

from .core import Candidate
from .trie import SIDTrie

Token = Hashable


def _as_sid(value: Any, field: str) -> tuple[Token, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON list.")
    if any(isinstance(token, (dict, list)) for token in value):
        raise ValueError(f"{field} tokens must be JSON scalars.")
    return tuple(value)


def _candidate(value: Any, field: str) -> Candidate:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object.")
    if not {"sid", "item_id", "direct_score"} <= set(value):
        raise ValueError(f"{field} requires sid, item_id, and direct_score.")
    return Candidate(_as_sid(value["sid"], f"{field}.sid"), value["item_id"], float(value["direct_score"]))


@dataclass(frozen=True)
class StaticLogitTable:
    """A JSON-serializable prefix-to-logits provider for a frozen rollout."""

    rows: dict[tuple[Token, ...], dict[Token, float]]

    @classmethod
    def from_records(cls, records: Any, field: str) -> "StaticLogitTable":
        if not isinstance(records, list):
            raise ValueError(f"{field} must be a list of prefix records.")
        rows: dict[tuple[Token, ...], dict[Token, float]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict) or "prefix" not in record or "logits" not in record:
                raise ValueError(f"{field}[{index}] requires prefix and logits.")
            prefix = tuple(record["prefix"])
            entries = record["logits"]
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"{field}[{index}].logits must be a non-empty list.")
            logits: dict[Token, float] = {}
            for entry in entries:
                if not isinstance(entry, dict) or "token" not in entry or "value" not in entry:
                    raise ValueError(f"{field}[{index}].logits entries require token and value.")
                token = entry["token"]
                if isinstance(token, (dict, list)) or token in logits:
                    raise ValueError(f"{field}[{index}] has an invalid or duplicate token.")
                logits[token] = float(entry["value"])
            if prefix in rows:
                raise ValueError(f"{field} repeats prefix {prefix!r}.")
            rows[prefix] = logits
        return cls(rows)

    def __call__(self, prefix: tuple[Token, ...]) -> dict[Token, float]:
        try:
            return self.rows[prefix]
        except KeyError as error:
            raise KeyError(f"No frozen route logits supplied for prefix {prefix!r}.") from error


@dataclass(frozen=True)
class RolloutInput:
    """One native beam plus frozen direct and reciprocal route observations."""

    anchor: Candidate
    beam: tuple[Candidate, ...]
    resolved_sids: dict[tuple[Token, ...], Candidate]
    direct_logits: StaticLogitTable
    reciprocal_logits: StaticLogitTable

    @classmethod
    def from_json(cls, value: Any) -> "RolloutInput":
        if not isinstance(value, dict):
            raise ValueError("rollout must be an object.")
        anchor = _candidate(value.get("anchor"), "anchor")
        raw_beam = value.get("beam")
        if not isinstance(raw_beam, list) or not raw_beam:
            raise ValueError("beam must be a non-empty list.")
        raw_resolved = value.get("resolved_sids")
        if not isinstance(raw_resolved, list) or not raw_resolved:
            raise ValueError("resolved_sids must list native item IDs and direct scores for possible repairs.")
        resolved = {_candidate(candidate, f"resolved_sids[{index}]").sid: _candidate(candidate, f"resolved_sids[{index}]") for index, candidate in enumerate(raw_resolved)}
        if len(resolved) != len(raw_resolved):
            raise ValueError("resolved_sids repeats a SID.")
        return cls(
            anchor=anchor,
            beam=tuple(_candidate(candidate, f"beam[{index}]") for index, candidate in enumerate(raw_beam)),
            resolved_sids=resolved,
            direct_logits=StaticLogitTable.from_records(value.get("direct_routes"), "direct_routes"),
            reciprocal_logits=StaticLogitTable.from_records(value.get("reciprocal_routes"), "reciprocal_routes"),
        )

    def validate(self, trie: SIDTrie) -> None:
        if not trie.is_legal(self.anchor.sid):
            raise ValueError("anchor.sid is not a terminal legal SID.")
        if self.anchor not in self.beam:
            raise ValueError("beam must include the stated anchor unchanged.")
        for candidate in self.beam:
            if not trie.is_legal(candidate.sid):
                raise ValueError(f"beam contains an illegal SID: {candidate.sid!r}")
            resolved = self.resolved_sids.get(candidate.sid)
            if resolved != candidate:
                raise ValueError("resolved_sids must preserve each beam candidate's native item ID and direct score.")
        for candidate in self.resolved_sids.values():
            if not trie.is_legal(candidate.sid):
                raise ValueError(f"resolved_sids contains an illegal SID: {candidate.sid!r}")
        prefix: tuple[Token, ...] = ()
        for token in self.anchor.sid:
            legal = set(trie.legal_children(prefix))
            for name, provider in (("direct", self.direct_logits), ("reciprocal", self.reciprocal_logits)):
                supplied = set(provider(prefix))
                if not legal <= supplied:
                    raise ValueError(f"{name} route omits legal children at prefix {prefix!r}.")
            prefix = (*prefix, token)

    def native_direct_score(self, sid: tuple[Token, ...]) -> float:
        try:
            return self.resolved_sids[sid].direct_score
        except KeyError as error:
            raise KeyError(f"No native direct score supplied for repaired SID {sid!r}.") from error

    def resolve_item(self, sid: tuple[Token, ...]) -> Token:
        try:
            return self.resolved_sids[sid].item_id
        except KeyError as error:
            raise KeyError(f"No native item ID supplied for repaired SID {sid!r}.") from error


def trie_from_json(value: Any) -> SIDTrie:
    if not isinstance(value, list) or not value:
        raise ValueError("trie JSON must be a non-empty list of SID paths.")
    return SIDTrie(_as_sid(path, f"trie[{index}]") for index, path in enumerate(value))
