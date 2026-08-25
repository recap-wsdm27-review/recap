# Backbone adapter contract

An adapter is the only place where RECAP meets a particular multimodal generator. It must preserve the model's native decoding and evaluation protocol.

## Required read-only inputs

For each output SID modality, expose:

1. a direct-logit callable and reciprocal-logit callable over the same output vocabulary;
2. logits queried under the candidate's realized prefix, never the target SID prefix at test time;
3. the exact trie/legal-mask and collision-resolution policy used by the backbone;
4. the native direct sequence score, including length normalization if any;
5. a SID-to-item resolver and the unchanged downstream fusion function.

## Required training and validation boundary

The adapter may train a locator and learned repairer only from training-split rollouts. It must record calibration data, validation-selected thresholds, mixing weight, stopping policy, split hashes, checkpoint fingerprints, and random seeds before test evaluation. Test examples cannot select thresholds, variants, or repair rounds.

## Prohibited adapter behavior

An adapter must not widen the final beam, add critic/locator/actionability terms to final item scores, use target-side modalities or labels at inference, change fusion, or silently replace the backbone's SID mapping. A reciprocal task participates in RECAP only when it supplies same-output-space logits under the realized prefix.

## Minimal adapter validation

Before claiming a bound adapter, verify that: legal probabilities sum to one; direct and reciprocal support have identical legal keys; every repair is a terminal legal SID; reintegration retains native width; and public manifests identify source revision, data version, split hashes, and checkpoints without exposing private paths or credentials.
