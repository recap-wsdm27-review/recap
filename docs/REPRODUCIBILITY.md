# Reproducibility boundary

The repository currently supports method-level verification, not benchmark reproduction. A future experiment release must add a versioned manifest containing the following, before any table or figure is described as reproduced:

- source revision and license for each bound backbone adapter;
- raw-data source/version, preprocessing command, and SHA-256 hashes of processed splits;
- SID tokenizer, trie, collision-policy, fusion, and checkpoint fingerprints;
- training-only rollout cache manifest and locator/repairer configuration;
- validation-selected temperatures, thresholds, mixing weight, stopping rule, and seed policy;
- evaluation command, exact candidate protocol, per-seed raw outputs, aggregation script, and environment details.

The manifest should permit a reviewer to recompute results without publishing private paths, credentials, raw user logs, or checkpoints that cannot legally be redistributed. Until those inputs exist, `pytest -q` verifies only the public method invariants.
