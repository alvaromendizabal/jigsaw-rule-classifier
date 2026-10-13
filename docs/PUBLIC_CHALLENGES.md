# Evaluate the boundary of lexical policy matching

The public demo is useful because its reasoning can be inspected. This evaluation makes its failure boundary inspectable too: **24 visible authored cases, 16 expected outcomes matched, 8 diagnostic failures retained**. It executes the shipped TF–IDF engine with its unchanged support examples and configuration. It does not evaluate the retained neural model.

```bash
node tools/evaluate_public_challenges.mjs --check
node tools/test_public_challenges.mjs
```

Use Node.js 24.19.0, the CI runtime. No installation, credentials, model download, or cloud compute is required. Open [`reports/public_challenges.html`](../reports/public_challenges.html) locally for the responsive, printable review. The [JSON report](../reports/public_challenges.json) contains every outcome, metric denominator, policy slice, and source SHA-256 digest. The [authored inputs](../examples/public_challenges.json) are fully public.

## What the cases demonstrate

| Behavior | Matches / cases | Engineering interpretation |
|---|---:|---|
| Direct lexical matches | 8 / 8 | The exposed similarity mechanism behaves as intended on close wording. |
| Policy conditioning | 3 / 3 | The same sales pitch receives different outcomes under different policies. |
| Unicode/case normalization | 2 / 2 | Width and case variants retain the intended behavior. |
| Unrelated vocabulary | 2 / 2 | Unsupported inputs abstain. |
| Quotation | 1 / 2 | Similarity can mistake an analyzed sales quotation for an actual promotion. |
| Negation | 0 / 3 | Surface overlap outweighs the negated intent in these examples. |
| Explicit exception | 0 / 1 | A spoiler warning does not reliably override matching plot terms. |
| Paraphrase | 0 / 3 | Three semantically relevant violations abstain without familiar vocabulary. |

The five false violation matches and three paraphrase abstentions are preserved; the engine was **not tuned against these cases**. On the 22 authored allowed/violation cases it makes 19 decisions. Fourteen match their label: **14/19 selective agreement**, distinct from **16/24 agreement over the entire suite**, which also includes two expected abstentions. These counts demonstrate why support-adapted semantic models and policy-transfer evaluation matter; they do not measure those models' quality.

## Evaluation contract

Expected labels are authored interpretations of the fictional policies. They are visible, deliberately selected, and not a random or untouched sample. The suite includes existing demo-style inputs and new variations. Do not infer statistical uncertainty, generalization, moderation safety, or a new historical score from these counts.

Only the policy and query enter `analyze()`. Labels enter the evaluator afterward. Balanced or unsupported engine states map to abstention. The report keeps wrong decisions and abstentions separate, and reports zero-denominator rates as `null`. Per-policy and per-behavior summaries use case counts rather than averaging unequal slice percentages.

`--check` recomputes both saved reports and fails if an outcome, source identity, or HTML byte differs. SHA-256 covers the evaluator, adapter, authored suite, and actual engine. Source identity includes comments; an unchanged prediction does not hide changed source. The employer review additionally fingerprints all declared demo assets, challenge tests, and published reports. This is an explicit review surface, not a claim to hash every transitive repository dependency.

For an intentional revision, inspect the case and source changes, run `node tools/evaluate_public_challenges.mjs --write`, review both report diffs, and run the checks again. The update flag is explicit; normal invocation prints JSON without writing. CI checks results and failure paths and retains the two reports as artifacts. Historical neural evidence, notebooks, source credits, and private research remain unchanged.
