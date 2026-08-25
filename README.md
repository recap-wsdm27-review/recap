# RECAP: Reciprocal Evidence Critique with Actionable Prefix-repair

Anonymous supplementary implementation for the WSDM 2027 submission *Beyond Late Fusion: Reciprocal Critique and Revocable Decoding for Multimodal Generative Recommendation*.

RECAP is an inference-time plug-in for a **frozen** multimodal generative recommender with hierarchical semantic identifiers (SIDs). It does not replace the backbone, widen its final candidate budget, or add a new final ranking score. Instead, it audits the best direct trajectory in each output SID space with reciprocal evidence conditioned on the other modality, selectively repairs a disputed suffix inside the legal trie, and returns the proposal to the original direct-score beam.

## Design and release scope

This is a dependency-light, backbone-agnostic implementation of the RECAP decoding layer. It provides SID trie legality, legal-set route normalization, evidence features, actionability gating, prefix revocation, refreshed PoE repair, bounded refinement, fixed-budget reintegration, a portable rollout format, and command-line execution.

Backbone-specific systems provide their native direct and reciprocal routes, SID mapping, direct sequence scorer, and final fusion through the formal [adapter contract](docs/ADAPTER_CONTRACT.md). The [paper-to-code map](docs/PAPER_CODE_MAP.md) records the correspondence between manuscript components and the public modules.

## Method contract

For one modality's direct candidate `y`, RECAP needs two logit providers that both predict the **same output SID space**:

- `direct_logits(prefix)`: native direct-route logits at the realized prefix;
- `reciprocal_logits(prefix)`: opposite-view logits over that same output space and prefix;
- a `SIDTrie` encoding the backbone's legal child set and collision policy;
- the backbone's unchanged direct sequence score and SID-to-item resolver.

At each level, both logits are normalized only over legal children of the current direct prefix. The locator may use committed-token support, top-two margins, normalized entropies, Jensen-Shannon disagreement, committed ranks, and shared-alternative agreement. A gate accepts a proposed first-error level only when locator confidence, actionable log-PoE margin, and normalized reciprocal entropy pass validation-frozen thresholds; singleton legal sets always abstain.

If accepted, RECAP keeps the prefix preceding the proposed level and regenerates the complete suffix. `PoERepairer` re-queries both routes after every generated token, preventing evidence from the revoked branch from being reused. The best legal proposal is deduplicated and ranked only by the original direct score within the original beam width.

## Install and test

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The package itself has no runtime dependencies beyond Python 3.10+. `pytest` is used only for the test suite.

## Minimal integration sketch

```python
from recap import Candidate, PoERepairer, RecapEngine, SIDTrie

trie = SIDTrie(catalog_sid_paths)
engine = RecapEngine(
    trie=trie,
    locator=trained_locator,       # returns (first_error_level | None, confidence)
    repairer=PoERepairer(direct_logits, reciprocal_logits, mixing_weight=0.5),
    error_threshold=validation_tau_error,
    action_threshold=validation_tau_action,
    max_reciprocal_entropy=validation_tau_entropy,
)

anchor = Candidate(native_top_sid, native_item_id, native_direct_score)
result = engine.refine(
    anchor, direct_logits, reciprocal_logits,
    native_direct_score, sid_to_item, max_rounds=2,
)
updated_beam = engine.reintegrate(native_beam, result.candidate, width=len(native_beam))
```

The sketch leaves calibration, training, validation selection, and native fusion in the owning backbone adapter. Do not substitute test labels or target SIDs for any inference-time input.

## Repository guide

- `src/recap/`: dependency-light RECAP primitives.
- `tests/`: unit tests for legality, gate behavior, refreshed PoE repair, and beam-budget preservation.
- `docs/PAPER_CODE_MAP.md`: equation/section-level implementation map.
- `docs/ADAPTER_CONTRACT.md`: required boundary for a real backbone adapter.
- `docs/REPRODUCIBILITY.md`: release inputs required before result reproduction can be claimed.
- `docs/ANONYMITY_AUDIT.md`: review-time identity and artifact hygiene.
- `docs/RUNNABLE_WORKFLOW.md`: portable JSON rollout format and command-line workflow.

## Anonymity and release hygiene

No author names, affiliations, email addresses, personal URLs, machine paths, private service identifiers, raw data, checkpoints, or manuscript result registries belong in this repository. Before publishing an update, run the checks in [docs/ANONYMITY_AUDIT.md](docs/ANONYMITY_AUDIT.md) and confirm that the public commit history contains no inherited source history.
