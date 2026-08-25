# Paper-to-code map

This map records the public implementation boundary for the current manuscript. It intentionally does not reproduce benchmark numbers in source code.

| Manuscript component | Public implementation | Status |
| --- | --- | --- |
| Sec. 3.1: direct and reciprocal routes in one output SID space | `RecapEngine.evidence_for` accepts separate direct/reciprocal providers at the same realized prefix | Implemented core; route construction belongs to the adapter |
| Sec. 3.2, Eqs. 3-4: legal-set normalization | `_softmax_on_legal` and `SIDTrie.legal_children` | Implemented core |
| Sec. 4.1, Eq. 7: route-evidence features | `RouteEvidence.features` | Implemented feature extraction; no learned/calibrated locator weights are shipped |
| Sec. 4.1, Eq. 8: first-error labels | Adapter-side training target | Defined by the adapter training protocol |
| Sec. 4.2, Eq. 9: actionability and abstention | `RouteEvidence.actionability_margin`, `RecapEngine.decide` | Implemented core; threshold selection is adapter-side validation work |
| Sec. 4.3, Eq. 10: suffix revocation | `RecapEngine.revise` | Implemented core |
| Sec. 4.4, Eq. 11: PoE repair | `PoERepairer` | Implemented and refreshed at every generated prefix |
| Sec. 4.4, Eq. 12: learned repairer | `Repairer` protocol | Adapter-facing learned repair interface |
| Sec. 4.4: at most two refreshed rounds | `RecapEngine.refine(max_rounds=2)` | Implemented control flow; real providers required |
| Sec. 4.5: direct-score, fixed-slot reintegration | `RecapEngine.reintegrate` | Implemented core |
| Secs. 5-6: three backbones, Amazon data, paired seeds, tables/figures | Artifact manifest and adapters | Experiment execution and reporting protocol |

## Evaluation boundary

Benchmark evaluation, timing, seed-level aggregation, and table/figure generation are governed by the versioned artifact manifest in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The public source keeps these operations separate from the backbone-agnostic decoding layer so that native scoring, fusion, data processing, and evaluation remain under their owning adapter.
