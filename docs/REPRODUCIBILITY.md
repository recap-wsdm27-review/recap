# Reproducibility manifest

Each experiment run is represented by a versioned manifest containing:

- source revision and license for each bound backbone adapter;
- raw-data source/version, preprocessing command, and SHA-256 hashes of processed splits;
- SID tokenizer, trie, collision-policy, fusion, and checkpoint fingerprints;
- training-only rollout cache manifest and locator/repairer configuration;
- validation-selected temperatures, thresholds, mixing weight, stopping rule, and seed policy;
- evaluation command, exact candidate protocol, per-seed raw outputs, aggregation script, and environment details.

The manifest permits a reviewer to recompute results without publishing private paths, credentials, raw user logs, or checkpoints that cannot legally be redistributed. `pytest -q` verifies the public method invariants, while the manifest binds those invariants to native model, data, and evaluation artifacts.
