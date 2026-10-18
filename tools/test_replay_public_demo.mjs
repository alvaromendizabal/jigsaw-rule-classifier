import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { generate, replay, options } from "./replay_public_demo.mjs";
import { runBundle, readJson, digest, match } from "./replay_bundle.mjs";
import { PRESETS, exportResult } from "../public-demo/engine.js";

test("the actual browser export is independently recomputed for every context", () => {
  const exported = generate();
  assert.equal(replay(exported).valid_contexts, 3);
  assert.equal(replay(exported).active_state, "VIOLATION_MATCH");
  const edited = structuredClone(PRESETS);
  edited[0].rule += " Private sales are allowed when explicitly requested.";
  const result = exportResult(edited, "a local edited query with notebook paper", "promotion");
  assert.equal(replay(result).valid_contexts, 3);
});
test("query copies are purged during replay and invalid inactive contexts stay rejected", () => {
  const contexts = structuredClone(PRESETS);
  const query = contexts[0].allowed[0];
  contexts[1].violation = [];
  const report = replay(exportResult(contexts, query, "promotion"));
  assert.equal(report.purged_query_support_rows, 1);
  assert.equal(report.valid_contexts, 2);
  assert.equal(report.rejected_contexts, 1);
});
for (const [name, mutate] of [
  ["invented score", x => x.comparisons[0].result.margin += 0.01],
  ["invented explanation", x => x.comparisons[0].result.evidence[0].contributions[0].contribution += 0.01],
  ["changed decision", x => x.comparisons[0].result.state = "ALLOWED_MATCH"],
  ["changed input", x => x.query = "a completely different query"],
  ["changed rule", x => x.contexts[0].rule += " Except purchases."],
  ["changed engine config", x => x.engine.ruleBoost = 99],
  ["missing comparison", x => x.comparisons.pop()],
  ["unknown output", x => x.private_accuracy = 0.99],
  ["unknown context field", x => x.contexts[0].hidden_label = 1],
  ["duplicate context", x => x.contexts.push(x.contexts[0])],
  ["missing active context", x => x.activeContext = "missing"],
  ["oversized query", x => x.query = "a".repeat(1201)],
  ["invalid active context", x => x.contexts[0].violation = []],
]) test(`replay rejects ${name}`, () => {
  const changed = structuredClone(generate()); mutate(changed);
  assert.throws(() => replay(changed));
});
test("object order is irrelevant and numeric tolerance never accepts changed labels or counts", () => {
  const exported = generate();
  assert.equal(replay(Object.fromEntries(Object.entries(exported).reverse())).valid_contexts, 3);
  const numeric = structuredClone(exported);
  numeric.comparisons[0].result.margin += 1e-14;
  assert.equal(replay(numeric).valid_contexts, 3);
  assert.throws(() => match(3, 3.00000000000001, "count", 1e-12));
  assert.throws(() => match(0.5, NaN, "score", 1e-12));
});
function fixture(run) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), "policy-replay-"));
  try { run(temp, path.join(temp, "output")); }
  finally { fs.rmSync(temp, { recursive: true, force: true }); }
}
test("create, imported-input replay and verification preserve byte-identical artifacts", () => fixture((temp, output) => {
  runBundle(options, ["--output", output]);
  const before = fs.readdirSync(output).map(name => digest(fs.readFileSync(path.join(output, name))));
  runBundle(options, ["--output", output, "--check"]);
  assert.deepEqual(fs.readdirSync(output).map(name => digest(fs.readFileSync(path.join(output, name)))), before);
  runBundle(options, ["--input", path.join(output, "export.json"), "--output", path.join(temp, "imported")]);
  assert.deepEqual(readJson(path.join(temp, "imported/export.json")), readJson(path.join(output, "export.json")));
}));
test("recomputed checksums cannot hide fabricated model output", () => fixture((_, output) => {
  runBundle(options, ["--output", output]);
  const file = path.join(output, "export.json");
  const changed = readJson(file); changed.comparisons[0].result.margin += 0.25;
  fs.writeFileSync(file, JSON.stringify(changed));
  const manifest = readJson(path.join(output, "manifest.json"));
  manifest.files["export.json"] = digest(fs.readFileSync(file));
  fs.writeFileSync(path.join(output, "manifest.json"), JSON.stringify(manifest));
  assert.throws(() => runBundle(options, ["--output", output, "--check"]), /Replay mismatch/);
}));
test("source-only changes invalidate receipt reuse", () => fixture((temp, output) => {
  const sourceRoot = path.join(temp, "source");
  for (const name of ["public-demo/engine.js", "public-demo/app.js", "tools/replay_bundle.mjs", "tools/replay_public_demo.mjs"]) {
    fs.mkdirSync(path.dirname(path.join(sourceRoot, name)), { recursive: true });
    fs.copyFileSync(path.join(options.root, name), path.join(sourceRoot, name));
  }
  const local = { ...options, root: sourceRoot };
  runBundle(local, ["--output", output]);
  fs.appendFileSync(path.join(sourceRoot, "public-demo/engine.js"), "\n// changed source\n");
  assert.throws(() => runBundle(local, ["--output", output, "--check"]), /source_sha256/);
}));
test("malformed, symlinked and oversized imports never publish a success bundle", () => fixture((temp, output) => {
  const input = path.join(temp, "input.json");
  fs.writeFileSync(input, "{malformed");
  assert.throws(() => runBundle(options, ["--input", input, "--output", output]));
  assert.equal(fs.existsSync(output), false);
  const link = path.join(temp, "link.json"); fs.symlinkSync(input, link);
  assert.throws(() => runBundle(options, ["--input", link, "--output", output]), /Symlink/);
  fs.writeFileSync(input, "x".repeat(8 * 1024 * 1024 + 1));
  assert.throws(() => runBundle(options, ["--input", input, "--output", output]), /8 MiB/);
  assert.equal(fs.existsSync(output), false);
}));
test("unknown flags, inside-checkout outputs and missing verification artifacts fail", () => fixture((_, output) => {
  assert.throws(() => runBundle(options, ["--invent-pass"]));
  assert.throws(() => runBundle(options, ["--output", path.join(options.root, "generated")]));
  assert.throws(() => runBundle(options, ["--output", output, "--check"]));
}));
