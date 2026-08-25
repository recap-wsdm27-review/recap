# Anonymous release audit

Run this audit before every review-period push.

1. Inspect public repository, organization, and commit URLs while logged out.
2. Confirm public commit authors, commit messages, repository description, organization profile, and public members contain no author or institution clues.
3. Scan tracked text for author names, affiliations, email addresses, personal handles, local paths, private hostnames, service IDs, acknowledgements, and credentials.
4. Confirm no inherited Git history is pushed from a personal or institutional source repository.
5. Exclude raw data, caches, checkpoints, artifacts with embedded metadata, and generated PDFs unless they have separately passed the same audit.
6. Re-check every external URL in the manuscript as an unauthenticated visitor.

The public tree must contain only review-safe source, tests, documentation, and explicitly redistributable configuration. Removing a filename from the working tree is not enough: inspect the public commit history and GitHub metadata as well.
