# Paper-to-code scope

## Implemented core

| Paper component | Public module | Invariant enforced |
| --- | --- | --- |
| Legal child set \(\mathcal A_m(y_{<\ell})\) | `recap.trie.SIDTrie` | Every repaired token remains in the catalog trie. |
| Legal-set-normalized direct/reciprocal beliefs | `RecapEngine.evidence_for` | Both routes are normalized over the same realized prefix and legal children. |
| First-error/abstention separation | `Locator` + `RecapEngine.decide` | A proposed level is not acted on unless confidence, actionability, and reciprocal certainty pass frozen thresholds. |
| Prefix-preserving revocation | `RecapEngine.revise` | Only the suffix from the accepted level is supplied to the repairer. |
| Refreshed PoE repair and bounded refinement | `PoERepairer`, `RecapEngine.refine` | Each regenerated token re-queries both routes under its new prefix; later rounds cannot move before the accepted level. |
| Direct-score-only reintegration | `RecapEngine.reintegrate` | Candidate scores, deduplication, and beam width remain native. |
| Portable frozen-rollout validation | `recap.rollout`, `python -m recap validate` | A release can verify trie, beam, route-logit, and native-resolution consistency without a dataset. |

## Required adapter contract

Each backbone adapter must expose, for an existing direct candidate prefix:

1. direct logits and reciprocal logits over the same output SID space;
2. the native trie/collision policy and SID-to-item resolver;
3. the backbone's direct sequence score and unchanged final fusion; and
4. train/validation/test split hashes plus frozen checkpoint and SID fingerprints.

## Deliberately absent until verified remote source is bound

- MQL4GRec, MACRec, and SynGR adapters;
- Amazon processed data, images, checkpoints, caches, and predictions;
- any fitted/manuscript-only result registry or synthetic seed fixture; and
- paper result claims, latency values, or reproduced tables.

The next implementation step is to bind a read-only snapshot of the actual remote source, then add adapters and an artifact manifest without changing the paper's experiment-selection boundary.
