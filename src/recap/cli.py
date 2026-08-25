"""Command-line entry points for validating and exercising frozen rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import PoERepairer, RecapEngine
from .rollout import RolloutInput, trie_from_json


def _load_json(path: str) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_inputs(args: argparse.Namespace):
    trie = trie_from_json(_load_json(args.trie))
    rollout = RolloutInput.from_json(_load_json(args.rollout))
    rollout.validate(trie)
    return trie, rollout


def _candidate_json(candidate):
    return {"sid": list(candidate.sid), "item_id": candidate.item_id, "direct_score": candidate.direct_score}


def validate(args: argparse.Namespace) -> int:
    trie, rollout = _load_inputs(args)
    print(json.dumps({"valid": True, "sid_depth": len(rollout.anchor.sid), "beam_width": len(rollout.beam), "trie_root_children": len(trie.legal_children(()))}))
    return 0


def run_poe(args: argparse.Namespace) -> int:
    trie, rollout = _load_inputs(args)
    locator = lambda _evidence: (args.first_error_level, args.locator_confidence)
    engine = RecapEngine(
        trie=trie,
        locator=locator,
        repairer=PoERepairer(rollout.direct_logits, rollout.reciprocal_logits, mixing_weight=args.mixing_weight),
        error_threshold=args.error_threshold,
        action_threshold=args.action_threshold,
        max_reciprocal_entropy=args.max_reciprocal_entropy,
        mixing_weight=args.mixing_weight,
    )
    result = engine.refine(
        rollout.anchor,
        rollout.direct_logits,
        rollout.reciprocal_logits,
        rollout.native_direct_score,
        rollout.resolve_item,
        max_rounds=args.max_rounds,
    )
    updated = engine.reintegrate(rollout.beam, result.candidate, width=len(rollout.beam))
    print(json.dumps({
        "candidate": _candidate_json(result.candidate),
        "decisions": [decision.__dict__ for decision in result.decisions],
        "beam": [_candidate_json(candidate) for candidate in updated],
    }, default=str))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or execute a frozen RECAP rollout.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--trie", required=True, help="JSON list of terminal SID paths")
    common.add_argument("--rollout", required=True, help="JSON native beam and prefix-indexed route logits")
    validate_parser = subcommands.add_parser("validate", parents=[common], help="validate a portable rollout")
    validate_parser.set_defaults(handler=validate)
    run_parser = subcommands.add_parser("run-poe", parents=[common], help="execute refreshed PoE repair")
    run_parser.add_argument("--first-error-level", required=True, type=int, help="zero-based locator output")
    run_parser.add_argument("--locator-confidence", required=True, type=float)
    run_parser.add_argument("--error-threshold", required=True, type=float)
    run_parser.add_argument("--action-threshold", required=True, type=float)
    run_parser.add_argument("--max-reciprocal-entropy", required=True, type=float)
    run_parser.add_argument("--mixing-weight", required=True, type=float)
    run_parser.add_argument("--max-rounds", required=True, type=int)
    run_parser.set_defaults(handler=run_poe)
    args = parser.parse_args()
    return args.handler(args)
