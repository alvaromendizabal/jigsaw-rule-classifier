# Replay a Policy Lens analysis

Policy Lens exports the edited rule, comment, support examples and computed results.
The replay command uses those inputs with the shipped browser engine to recompute
the analysis. It verifies every context, including rejected inactive contexts;
the active context must have a valid analysis.

## Run locally

From the repository root with Node.js 22 or 24:

```bash
node tools/replay_public_demo.mjs --input policy-lens-local-analysis.json --output /tmp/policy-replay
node tools/replay_public_demo.mjs --output /tmp/policy-replay --check
```

Omit `--input` to create a reference analysis using the authored fictional presets.
On Windows, select a writable directory outside the checkout. Create its parent
directory first. A fresh output directory is published only after replay succeeds.
Existing bundles are verified, never silently overwritten.

| Bundle member | Purpose |
|---|---|
| `export.json` | The exact supplied editable inputs and browser outputs |
| `receipt.json` | Recomputed analysis scope, checks, source hashes, UTC time and runtime |
| `manifest.json` | SHA-256 identities of the export and receipt |

An export includes your edited text. Files stay local; this command has no model
download, upload, remote inference or account dependency.

## What is recomputed

1. Support-only TF–IDF vocabulary, duplicate handling and exact query-overlap purging.
2. Every valid context's similarities, selected supports, token contributions and margin.
3. Abstention states and the exact validation errors for invalid inactive contexts.
4. Active context identity, engine configuration and the complete export shape.

Noninteger numeric values use a bound of `1e-12 * (1 + abs(expected))` to allow
last-bit differences between browser and Node implementations of `Math.log`.
Integer counts, text, decisions, identifiers and field inventories must match
exactly. NaN and infinite values fail. This is a numerical replay tolerance,
not permission for a changed model decision.

The bundle fingerprints `public-demo/engine.js`, `public-demo/app.js` and both replay
modules. Source changes invalidate reuse even if predictions happen to match.
The wider [public review](../review/README.md) also fingerprints the full declared
demo/tool source inventory, including additions and deletions.

## Failure behavior

```bash
node --test tools/test_replay_public_demo.mjs
```

The regression suite covers edited inputs, invalid contexts, overlap purging,
fabricated scores and explanations, changed decisions, source drift, malformed
JSON, oversized imports, symlink paths, field additions, missing comparisons,
unknown flags and verification of missing bundles. Recomputing an artifact hash
cannot hide fabricated output because the engine runs again during verification.

The transport reuses the same owner's public Kaggriculture replay implementation;
Policy Lens has its own input schema and domain checks.

## Evidence scope

A successful replay establishes consistency with this public lexical engine.
It does not establish that a moderation judgment is correct, calibrate a probability,
reproduce Qwen inference, or independently authenticate a historical score.
Starter examples are fictional; exported edits are labeled `EDITABLE_LOCAL_TEXT`
because their content is chosen by the person using the demo.
