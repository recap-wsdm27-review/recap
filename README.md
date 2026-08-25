# RECAP: Reciprocal Evidence Critique with Actionable Prefix-repair

Anonymous supplementary code for the WSDM 2027 submission *Beyond Late Fusion: Reciprocal Critique and Revocable Decoding for Multimodal Generative Recommendation*.

RECAP is a frozen-backbone inference plug-in for hierarchical semantic identifiers (SIDs). It audits a direct decoded trajectory with a reciprocal route in the **same output SID space**, selectively revokes the suffix after a likely first divergence, repairs only inside the reopened legal subtree, and reintegrates the proposal under the original beam budget and direct scoring rule.

## What is available now

The repository contains a dependency-light, runnable core implementation of:

- constrained SID tries and prefix legality checks;
- legal-set normalization of direct and reciprocal logits;
- pluggable first-error localization and conservative actionability gating;
- prefix-preserving suffix replacement; and
- direct-score-only, fixed-budget beam reintegration.

The paper-specific adapters for MQL4GRec, MACRec, and SynGR are intentionally not claimed to be included yet: the corresponding remote source tree has not been bound and audited in this repository. No fitted manuscript registry, synthetic result, checkpoint, raw dataset, or claimed benchmark result is included.

## Installation and test

```bash
python -m pip install -e '.[dev]'
pytest -q
```

## Design contract

An adapter supplies direct and reciprocal route logits under an existing candidate prefix, the backbone's direct sequence score, and the native SID-to-item resolver. RECAP does not change the backbone parameters, final fusion rule, or final beam size. See `docs/IMPLEMENTATION_SCOPE.md` for the paper-to-code mapping and the pending adapter boundary.

## Anonymity

Do not commit author names, affiliations, email addresses, personal URLs, acknowledgements, machine-specific paths, private service identifiers, or non-anonymous repository links during review.
