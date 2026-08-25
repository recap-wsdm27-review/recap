# Runnable workflow

The package includes a complete, data-agnostic path for executing a frozen rollout. It is designed for a bound backbone adapter to export one native beam and the exact route observations needed for review or debugging. It does not ship a dataset, learned locator, benchmark configuration, or numeric default.

## 1. Export adapter artifacts

Export three JSON documents from a frozen adapter:

- `trie.json`: a JSON list of terminal SID paths;
- `rollout.json`: one native beam, route-logit records, and the native scorer/resolver outputs for every SID that a repair may return;
- a separately versioned experiment configuration holding calibration and gate values selected on validation.

The rollout file has this structure (ellipses denote adapter-provided values, not repository defaults):

```json
{
  "anchor": {"sid": ["..."], "item_id": "...", "direct_score": "..."},
  "beam": [{"sid": ["..."], "item_id": "...", "direct_score": "..."}],
  "resolved_sids": [{"sid": ["..."], "item_id": "...", "direct_score": "..."}],
  "direct_routes": [
    {"prefix": [], "logits": [{"token": "...", "value": "..."}]}
  ],
  "reciprocal_routes": [
    {"prefix": [], "logits": [{"token": "...", "value": "..."}]}
  ]
}
```

`resolved_sids` is required because RECAP ranks a new proposal with the backbone's original direct score and resolves it with the original SID-to-item policy. A static export must therefore include those outputs for every repair candidate it permits. A live adapter can provide the same operations as callables instead.

## 2. Validate without selecting any parameter

```bash
python -m recap validate --trie trie.json --rollout rollout.json
```

This command checks terminal SID legality, beam/anchor consistency, availability of direct and reciprocal logits at every anchor prefix, and legal-child coverage. It is safe to run before training or validation choices are final.

## 3. Execute a validation-frozen PoE run

```bash
python -m recap run-poe \
  --trie trie.json --rollout rollout.json \
  --first-error-level <adapter-locator-output> \
  --locator-confidence <adapter-locator-confidence> \
  --error-threshold <validation-frozen-value> \
  --action-threshold <validation-frozen-value> \
  --max-reciprocal-entropy <validation-frozen-value> \
  --mixing-weight <validation-frozen-value> \
  --max-rounds <validation-frozen-value>
```

The reference command takes all numerical choices explicitly. It intentionally has no paper-derived defaults: thresholds, confidence calibration, evidence mixing, and stopping policy must be selected using validation data and recorded by the adapter. Its JSON output contains the proposal, every gate decision, and the fixed-width reintegrated beam.

## 4. Replace the reference locator only after training is bound

The command-line driver accepts a supplied first-error decision so that the rest of the method can be executed and audited before releasing learned weights. A complete backbone adapter should replace that input with its calibrated learned locator, retain the same `RecapEngine` contract, and record the train/validation provenance required by [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
