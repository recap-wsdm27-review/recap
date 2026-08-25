# Paper-to-code map

This map records what is implemented from the current manuscript and what remains an adapter or experiment obligation. It intentionally does not reproduce paper numbers or claim that a benchmark was run from this repository.

| Manuscript component | Public implementation | Status |
| --- | --- | --- |
| Sec. 3.1: direct and reciprocal routes in one output SID space | `RecapEngine.evidence_for` accepts separate direct/reciprocal providers at the same realized prefix | Implemented core; route construction belongs to the adapter |
| Sec. 3.2, Eqs. 3-4: legal-set normalization | `_softmax_on_legal` and `SIDTrie.legal_children` | Implemented core |
| Sec. 4.1, Eq. 7: route-evidence features | `RouteEvidence.features` | Implemented feature extraction; no learned/calibrated locator weights are shipped |
| Sec. 4.1, Eq. 8: first-error labels | Adapter-side training target | Not included; requires train-split rollouts and target SIDs |
| Sec. 4.2, Eq. 9: actionability and abstention | `RouteEvidence.actionability_margin`, `RecapEngine.decide` | Implemented core; threshold selection is adapter-side validation work |
| Sec. 4.3, Eq. 10: suffix revocation | `RecapEngine.revise` | Implemented core |
| Sec. 4.4, Eq. 11: PoE repair | `PoERepairer` | Implemented and refreshed at every generated prefix |
| Sec. 4.4, Eq. 12: learned repairer | `Repairer` protocol | Interface only; no invented learned weights or trainer |
| Sec. 4.4: at most two refreshed rounds | `RecapEngine.refine(max_rounds=2)` | Implemented control flow; real providers required |
| Sec. 4.5: direct-score, fixed-slot reintegration | `RecapEngine.reintegrate` | Implemented core |
| Secs. 5-6: three backbones, Amazon data, paired seeds, tables/figures | Artifact manifest and adapters | Not included until real source/data/artifacts are bound |

## Deliberate non-claims

The following are paper-level experimental claims and are not established by this supplementary core: benchmark quality, safety retention, timing, seed-level uncertainty, learned-capacity effects, adapter compatibility, or reproducibility of any table/figure. Their release requires the exact items in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
