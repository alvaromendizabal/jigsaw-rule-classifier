/** Public Policy Lens checks: pure numerical engine plus the real browser handlers. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(
  path.join(root, "public-demo/engine.js"),
  "utf8",
);
const engineURL = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const E = await import(engineURL);
const clone = (value) => JSON.parse(JSON.stringify(value));
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);
let count = 0;
async function test(name, fn) {
  await fn();
  console.log(`ok ${++count} - ${name}`);
}

await test("starter policies contain authored, balanced, distinct fictional examples", () => {
  assert.equal(E.PRESETS.length, 3);
  assert.equal(new Set(E.PRESETS.map((p) => p.id)).size, 3);
  for (const policy of E.PRESETS) {
    assert.equal(policy.violation.length, 4);
    assert.equal(policy.allowed.length, 4);
    assert.equal(
      new Set([...policy.violation, ...policy.allowed].map(E.canonical)).size,
      8,
    );
  }
});
await test("Unicode tokenization is stable and preserves explicit negation", () => {
  assert.deepEqual(E.tokens("The café is NOT a ＳＨＯＰ."), [
    "café",
    "not",
    "shop",
  ]);
  assert.equal(E.canonical(" Buy my course! "), E.canonical("BUY MY COURSE."));
  assert.equal(E.featureCounts("free tutorial free").get("free"), 2);
  assert.equal(E.featureCounts("free tutorial free").get("free tutorial"), 1);
});
await test("vocabulary and IDF fit support examples without using the query", () => {
  const p = E.PRESETS[0],
    a = E.analyze(p, "purple zeppelin buy course"),
    b = E.analyze(p, "unicorn telescope buy course");
  assert.equal(a.vocabularyFingerprint, b.vocabularyFingerprint);
  assert.equal(a.featureCount, b.featureCount);
  const context = E.prepareContext(p, "unseen query");
  const vocab = E.fitVocabulary(context.support, context.rule);
  assert.ok(!vocab.some((f) => f.term === "unseen" || f.term === "purple"));
  const term = vocab.find((f) => f.term === "course");
  const df = context.support.filter((s) =>
    E.featureCounts(s.text).has("course"),
  ).length;
  close(term.idf, Math.log(9 / (1 + df)) + 1);
});
await test("rule wording changes only declared feature weights and actual scores", () => {
  const first = {
    id: "test",
    name: "test",
    rule: "course",
    violation: ["buy course now", "buy lesson today"],
    allowed: ["free tutorial today", "free discussion now"],
  };
  const second = { ...first, rule: "free tutorial" };
  const a = E.analyze(first, "buy course free tutorial"),
    b = E.analyze(second, "buy course free tutorial");
  assert.notEqual(a.margin, b.margin);
  const context = E.prepareContext(first, "other");
  const vocab = E.fitVocabulary(context.support, "course");
  assert.equal(vocab.find((f) => f.term === "course").ruleBoost, 1.5);
  assert.equal(vocab.find((f) => f.term === "free").ruleBoost, 1);
});
await test("query copies are purged before fitting, including punctuation/case variants", () => {
  const policy = clone(E.PRESETS[0]);
  const query = policy.violation[0];
  const result = E.analyze(policy, query.toUpperCase().replace(".", "!"));
  assert.equal(result.purged, 1);
  assert.equal(result.supportCount, 7);
  assert.ok(
    !result.evidence.some((e) => E.canonical(e.text) === E.canonical(query)),
  );
});
await test("duplicate supports do not change fitted scores", () => {
  const policy = clone(E.PRESETS[0]),
    query = E.QUERIES[0].text;
  const original = E.analyze(policy, query);
  policy.violation.push(policy.violation[0]);
  const duplicate = E.analyze(policy, query);
  assert.equal(duplicate.duplicates, 1);
  assert.equal(duplicate.supportCount, 8);
  assert.equal(duplicate.margin, original.margin);
});
await test("conflicting labels and empty post-purge classes fail explicitly", () => {
  const policy = clone(E.PRESETS[0]);
  policy.allowed.push(policy.violation[0]);
  assert.throws(() => E.analyze(policy, "A query"), /conflicting labels/);
  assert.throws(
    () =>
      E.analyze(
        { rule: "rule", violation: ["buy it"], allowed: ["review it"] },
        "BUY IT!",
      ),
    /No violation examples remain/,
  );
});
await test("empty, oversized and malformed inputs fail before inference", () => {
  assert.throws(() => E.analyze(E.PRESETS[0], ""), /Comment cannot be empty/);
  assert.throws(() => E.analyze(E.PRESETS[0], "!"), /at least one word/);
  assert.throws(() => E.analyze(E.PRESETS[0], "a".repeat(1201)), /1200/);
  assert.throws(
    () => E.analyze({ ...E.PRESETS[0], rule: "" }, "query"),
    /Rule cannot be empty/,
  );
  assert.throws(
    () => E.analyze({ ...E.PRESETS[0], allowed: [] }, "query"),
    /1–8/,
  );
  assert.throws(
    () =>
      E.analyze({ ...E.PRESETS[0], violation: Array(9).fill("x") }, "query"),
    /1–8/,
  );
  assert.throws(
    () => E.analyze({ ...E.PRESETS[0], allowed: ["x".repeat(501)] }, "query"),
    /500/,
  );
});
await test("unknown wording produces no-evidence state rather than confidence", () => {
  const result = E.analyze(E.PRESETS[0], "xylophone zeppelin nebula");
  assert.equal(result.state, "NO_EVIDENCE");
  assert.equal(result.margin, 0);
  assert.equal(result.coverage, 0);
  assert.ok(!("probability" in result));
});
await test("each vector is zero or unit length and cosine explanations sum exactly", () => {
  const context = E.prepareContext(E.PRESETS[0], E.QUERIES[0].text),
    vocab = E.fitVocabulary(context.support, context.rule);
  const vector = E.vectorize(E.QUERIES[0].text, vocab);
  close(
    vector.values.reduce((s, v) => s + v * v, 0),
    1,
  );
  const result = E.analyze(E.PRESETS[0], E.QUERIES[0].text);
  for (const evidence of result.evidence) {
    close(
      evidence.contributions.reduce((s, f) => s + f.contribution, 0),
      evidence.similarity,
    );
    assert.ok(evidence.similarity >= 0 && evidence.similarity <= 1 + 1e-12);
  }
});
await test("margin equals the declared nearest-example contrast, not a probability", () => {
  const result = E.analyze(E.PRESETS[0], E.QUERIES[0].text);
  const avg = (label) =>
    result.evidence
      .filter((e) => e.label === label)
      .slice(0, 2)
      .reduce((s, e) => s + e.similarity, 0) / 2;
  close(result.margin, avg("violation") - avg("allowed"));
  assert.equal(result.selectedIds.length, 4);
  assert.match(result.interpretation, /not a probability/);
});
await test("same comment responds to context and exact input replay is deterministic", () => {
  const results = E.compare(E.PRESETS, E.QUERIES[0].text);
  assert.equal(results[0].result.state, "VIOLATION_MATCH");
  assert.equal(results[1].result.state, "ALLOWED_MATCH");
  assert.deepEqual(results, E.compare(E.PRESETS, E.QUERIES[0].text));
  assert.ok(results[0].result.margin > 0 && results[1].result.margin < 0);
});
await test("ambiguous lexical context produces a balanced state", () => {
  const result = E.analyze(
    {
      rule: "Discuss a topic",
      violation: ["red green"],
      allowed: ["blue green"],
    },
    "green",
  );
  close(result.margin, 0);
  assert.equal(result.state, "BALANCED");
});
await test("comparison isolates invalid contexts while preserving valid contexts", () => {
  const contexts = clone(E.PRESETS);
  contexts[1].rule = "";
  const results = E.compare(contexts, E.QUERIES[0].text);
  assert.ok(results[0].result);
  assert.equal(results[1].result, null);
  assert.match(results[1].error, /Rule/);
  assert.ok(results[2].result);
});
await test("export keeps edited inputs and honest starter/user-input scope", () => {
  const report = E.exportResult(E.PRESETS, E.QUERIES[0].text, "promotion");
  assert.equal(report.starterScope, "SYNTHETIC_ONLY");
  assert.equal(report.inputScope, "EDITABLE_LOCAL_TEXT");
  assert.equal(report.comparisons.length, 3);
  assert.match(report.purpose, /No competition data/);
  report.contexts[0].rule = "changed";
  assert.notEqual(report.contexts[0].rule, E.PRESETS[0].rule);
});

// The lightweight DOM below executes the real app's handlers and render code.
// It checks behavior and safe node construction, not browser layout or full a11y.
const html = fs.readFileSync(path.join(root, "public-demo/index.html"), "utf8");
const app = fs.readFileSync(path.join(root, "public-demo/app.js"), "utf8");
let loadCount = 0;
async function browser() {
  const nodes = new Map();
  let download = null,
    blob = null;
  class Element {
    constructor(tag = "div") {
      this.tag = tag;
      this.handlers = {};
      this.children = [];
      this.attributes = {};
      this.style = {};
      this.disabled = false;
      this.hidden = false;
      this.value = "";
      this._text = "";
    }
    set id(id) {
      this._id = id;
      nodes.set(id, this);
    }
    get id() {
      return this._id;
    }
    set textContent(text) {
      this._text = String(text);
      this.children = [];
    }
    get textContent() {
      return (
        this._text +
        this.children
          .map((c) => (typeof c === "string" ? c : c.textContent))
          .join("")
      );
    }
    append(...children) {
      this.children.push(...children);
    }
    replaceChildren(...children) {
      this._text = "";
      this.children = children;
      if (this.tag === "select") this.value = children[0]?.value || "";
    }
    setAttribute(name, value) {
      this.attributes[name] = value;
    }
    addEventListener(event, handler) {
      this.handlers[event] = handler;
    }
    click() {
      if (!this.disabled) this.handlers.click?.();
      if (this.download) download = this;
    }
  }
  for (const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const e = new Element(match[1]);
    e.id = match[2];
  }
  globalThis.document = {
    getElementById: (id) => nodes.get(id),
    createElement: (tag) => new Element(tag),
    createTextNode: (text) => text,
  };
  const originalCreate = URL.createObjectURL,
    originalRevoke = URL.revokeObjectURL;
  URL.createObjectURL = (value) => {
    blob = value;
    return "blob:public-test";
  };
  URL.revokeObjectURL = () => {};
  const module = app.replace(/(['"])\.\/engine\.js\1/, JSON.stringify(engineURL));
  assert.notEqual(module, app, "Test loader must replace the local engine import");
  await import(
    `data:text/javascript;base64,${Buffer.from(module).toString("base64")}#${++loadCount}`
  );
  return {
    nodes,
    get blob() {
      return blob;
    },
    get download() {
      return download;
    },
    restore() {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    },
    edit(id, value) {
      nodes.get(id).value = value;
      nodes.get(id).handlers.input();
    },
  };
}
await test("actual UI initially computes real results, evidence and policy comparisons", async () => {
  const b = await browser();
  assert.equal(b.nodes.get("result-status").textContent, "LOCAL RESULT");
  assert.equal(b.nodes.get("margin").textContent, "+0.286");
  assert.equal(b.nodes.get("evidence-list").children.length, 4);
  assert.equal(b.nodes.get("comparisons").children.length, 3);
  assert.equal(b.nodes.get("export").disabled, false);
  assert.ok(b.nodes.get("features").children.length > 0);
  b.restore();
});
await test("actual UI marks edited result/comparison stale and prevents stale exports", async () => {
  const b = await browser();
  b.edit("query", "A newly edited comment");
  assert.equal(b.nodes.get("result-status").textContent, "PREVIOUS RESULT");
  assert.equal(b.nodes.get("export").disabled, true);
  assert.ok(b.nodes.get("comparisons").children.every((c) => c.disabled));
  assert.match(
    b.nodes.get("comparison-status").textContent,
    /Previous comparison/,
  );
  b.restore();
});
await test("actual UI invalid input clears previous result and recovers after correction", async () => {
  const b = await browser();
  b.edit("rule", "");
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("margin").textContent, "—");
  assert.equal(b.nodes.get("evidence-list").children.length, 0);
  assert.equal(b.nodes.get("export").disabled, true);
  assert.equal(b.nodes.get("error").hidden, false);
  b.edit("rule", E.PRESETS[0].rule);
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("error").hidden, true);
  assert.equal(b.nodes.get("margin").textContent, "+0.286");
  b.restore();
});
await test("actual UI preserves edited policy examples across context switches", async () => {
  const b = await browser();
  const edited = E.PRESETS[0].rule + " Discuss purchases carefully.";
  b.edit("rule", edited);
  b.nodes.get("analyze").click();
  b.nodes.get("compare-attacks").click();
  assert.equal(b.nodes.get("policy").value, "attacks");
  assert.equal(b.nodes.get("query").value, E.QUERIES[0].text);
  b.nodes.get("compare-promotion").click();
  assert.equal(b.nodes.get("rule").value, edited);
  b.restore();
});
await test("actual UI inspects evidence, highlights real matches and toggles all supports", async () => {
  const b = await browser();
  b.nodes.get("show-all").click();
  assert.equal(b.nodes.get("evidence-list").children.length, 8);
  const row = b.nodes.get("evidence-list").children.at(-1);
  row.click();
  assert.equal(row.attributes["aria-pressed"], "false"); // Re-rendered node replaces this original button.
  assert.match(
    b.nodes.get("selected-description").textContent,
    /Compared with this/,
  );
  const selected = b.nodes
    .get("evidence-list")
    .children.find((e) => e.attributes["aria-pressed"] === "true");
  assert.ok(selected);
  b.nodes.get("show-all").click();
  assert.equal(b.nodes.get("evidence-list").children.length, 4);
  b.restore();
});
await test("actual UI example edits affect inference and query-overlap removal is visible", async () => {
  const b = await browser();
  b.edit("violation", E.PRESETS[0].allowed.join("\n"));
  b.edit("allowed", E.PRESETS[0].violation.join("\n"));
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("margin").textContent, "−0.286");
  b.nodes.get("reset").click();
  b.edit("query", E.PRESETS[0].violation[0]);
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("support-count").textContent, "7");
  assert.match(b.nodes.get("input-status").textContent, /exact query match/);
  b.edit(
    "allowed",
    E.PRESETS[0].allowed.join("\n") + "\n" + E.PRESETS[0].violation[0],
  );
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("export").disabled, true);
  assert.match(b.nodes.get("error").textContent, /conflicting/);
  b.restore();
});
await test("actual UI exports current computed JSON with editable text intact", async () => {
  const b = await browser();
  b.nodes.get("export").click();
  const report = JSON.parse(await b.blob.text());
  assert.equal(report.starterScope, "SYNTHETIC_ONLY");
  assert.equal(report.query, E.QUERIES[0].text);
  assert.equal(
    report.comparisons[0].result.margin,
    E.analyze(E.PRESETS[0], E.QUERIES[0].text).margin,
  );
  assert.equal(b.download.download, "policy-lens-local-analysis.json");
  b.restore();
});
await test("actual UI resets edits and treats HTML-like query text as text", async () => {
  const b = await browser();
  const query = "<img src=x onerror=alert(1)> buy my course";
  b.edit("query", query);
  b.nodes.get("analyze").click();
  assert.equal(b.nodes.get("highlighted-query").textContent, query);
  assert.ok(
    b.nodes
      .get("highlighted-query")
      .children.every((c) => typeof c === "string" || c.tag === "mark"),
  );
  b.nodes.get("reset").click();
  assert.equal(b.nodes.get("query").value, E.QUERIES[0].text);
  assert.equal(b.nodes.get("rule").value, E.PRESETS[0].rule);
  b.restore();
});
await test("public app uses local assets without uploads, persistence or neural claims", () => {
  assert.ok(
    !/\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\s*[.(]/.test(
      source + app,
    ),
  );
  assert.ok(!app.includes("innerHTML"));
  for (const match of html.matchAll(
    /<(?:script|link)\b[^>]*(?:src|href)="([^"]+)"/g,
  ))
    assert.ok(
      [
        "app.js",
        "styles.css",
        "https://alvaro-policy-lens.tartmacaw2.chatgpt.site/",
      ].includes(match[1]),
    );
  assert.match(html, /No Qwen model/);
  assert.match(html, /not a probability/);
});
console.log(
  `\n${count} Policy Lens checks passed. All starter text is authored synthetic data; no accuracy claim.`,
);
