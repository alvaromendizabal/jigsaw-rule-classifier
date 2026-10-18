/** Recompute a browser export from its exact editable inputs and the public engine. */
import path from "node:path";
import { fileURLToPath } from "node:url";
import { CONFIG, PRESETS, QUERIES, exportResult } from "../public-demo/engine.js";
import { match, requireThat, runBundle } from "./replay_bundle.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export function generate() {
  return exportResult(PRESETS, QUERIES[0].text, PRESETS[0].id);
}
export function replay(exported) {
  requireThat(exported?.schema === 1 && exported.starterScope === "SYNTHETIC_ONLY" &&
    exported.inputScope === "EDITABLE_LOCAL_TEXT", "Unsupported Policy Lens export.");
  requireThat(typeof exported.query === "string" && exported.query.trim().length > 0 &&
    exported.query.length <= CONFIG.maxQuery, "Invalid exported query.");
  requireThat(Array.isArray(exported.contexts) && exported.contexts.length >= 1 &&
    exported.contexts.length <= 16, "Expected 1–16 exported contexts.");
  const ids = new Set();
  for (const context of exported.contexts) {
    requireThat(context && typeof context === "object" && !Array.isArray(context),
      "Invalid policy context.");
    requireThat(Object.keys(context).every(key =>
      ["id", "name", "short", "rule", "violation", "allowed"].includes(key)),
    "Unexpected policy context fields.");
    requireThat(typeof context.id === "string" && /^[a-z][a-z0-9-]{0,63}$/.test(context.id) &&
      !ids.has(context.id), "Invalid or duplicate policy context ID.");
    ids.add(context.id);
    for (const key of ["name", "short"]) {
      if (key === "short" && !Object.hasOwn(context, key)) continue;
      requireThat(typeof context[key] === "string" && context[key].length > 0 &&
        context[key].length <= 200, "Invalid policy display name.");
    }
    requireThat(typeof context.rule === "string" && context.rule.length <= CONFIG.maxRule,
      "Invalid exported rule.");
    for (const label of ["violation", "allowed"]) {
      requireThat(Array.isArray(context[label]) && context[label].length <= CONFIG.maxPerLabel,
        "Invalid exported support collection.");
      requireThat(context[label].every(text => typeof text === "string" &&
        text.length <= CONFIG.maxExample), "Invalid exported support text.");
    }
  }
  requireThat(ids.has(exported.activeContext), "Active context is missing.");
  const expected = exportResult(exported.contexts, exported.query, exported.activeContext);
  const active = expected.comparisons.find(row => row.id === exported.activeContext);
  requireThat(active.result !== null, "Active context must have a valid analysis.");
  // Browser Math.log and Node Math.log may differ at their last representable bit.
  // Integers, IDs, source hashes, decisions, strings and object shapes remain exact.
  match(expected, exported, "export", 1e-12);
  const valid = expected.comparisons.filter(row => row.result !== null);
  return {
    scope: "EDITABLE_LOCAL_TEXT",
    engine: CONFIG.version,
    contexts: expected.contexts.length,
    valid_contexts: valid.length,
    rejected_contexts: expected.comparisons.length - valid.length,
    active_context: expected.activeContext,
    active_state: active.result.state,
    purged_query_support_rows: valid.reduce((sum, row) => sum + row.result.purged, 0),
    checks: [
      "support-only vocabulary, query-overlap purging and duplicate handling recomputed",
      "every context result or validation error recomputed",
      "margins, similarities, token contributions and abstention states recomputed",
      "source identity and complete export structure verified",
    ],
    tolerance: "Noninteger numbers: 1e-12 * (1 + abs(expected)); integers and all other fields exact.",
    interpretation: "Reproduces lexical analysis of editable text; no probability, correctness or neural-model claim.",
  };
}
export const options = { root, project: "jigsaw-rule-classifier", generate, replay };
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { runBundle(options); }
  catch (error) { console.error(`REPLAY_FAILED: ${error.message}`); process.exitCode = 1; }
}
