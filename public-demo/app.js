import {
  PRESETS,
  QUERIES,
  CONFIG,
  analyze,
  compare,
  exportResult,
  canonical,
} from "./engine.js";

const $ = (id) => document.getElementById(id);
const clone = (value) => JSON.parse(JSON.stringify(value));
const LABELS = {
  VIOLATION_MATCH: "Closer to violation examples",
  ALLOWED_MATCH: "Closer to allowed examples",
  BALANCED: "A mixed lexical match",
  NO_EVIDENCE: "Not enough lexical evidence",
};
let contexts = clone(PRESETS);
let activeId = contexts[0].id;
let current = null;
let comparisons = [];
let selectedId = null;
let showAll = false;
let dirty = false;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}
function signed(value) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}
function inputLines(id) {
  return $(id)
    .value.split(/\r?\n/)
    .map((text) => text.trim())
    .filter(Boolean);
}
function updateCounts() {
  $("character-count").textContent =
    `${$("query").value.length} / ${CONFIG.maxQuery}`;
  $("example-count").textContent =
    `${inputLines("violation").length + inputLines("allowed").length} examples`;
}
function storeEditor() {
  const context = contexts.find((item) => item.id === activeId);
  context.rule = $("rule").value;
  context.violation = inputLines("violation");
  context.allowed = inputLines("allowed");
}
function loadEditor() {
  const context = contexts.find((item) => item.id === activeId);
  $("policy").value = activeId;
  $("rule").value = context.rule;
  $("violation").value = context.violation.join("\n");
  $("allowed").value = context.allowed.join("\n");
  updateCounts();
}
function markDirty() {
  dirty = true;
  $("export").disabled = true;
  $("result-status").textContent = current ? "PREVIOUS RESULT" : "EDITED";
  $("result-status").className = "status stale";
  $("input-status").textContent =
    "Inputs changed. Analyze again to refresh the result and comparison.";
  $("comparison-status").textContent =
    "Previous comparison shown below. Analyze again to use the edited inputs.";
  $("error").hidden = true;
  updateCounts();
  renderComparisons();
}
function renderComparisons() {
  $("comparisons").replaceChildren(
    ...comparisons.map((item) => {
      const button = node(
        "button",
        `comparison-item${item.id === activeId ? " active" : ""}`,
      );
      button.id = `compare-${item.id}`;
      button.disabled = dirty;
      button.setAttribute("aria-pressed", String(item.id === activeId));
      button.setAttribute(
        "aria-label",
        `${item.name}: ${item.error ? item.error : LABELS[item.result.state] + ", lexical margin " + signed(item.result.margin)}. Inspect this context.`,
      );
      const top = node("span", "comparison-top");
      top.append(
        node("span", "", item.name),
        node(
          "span",
          "comparison-value",
          item.result ? signed(item.result.margin) : "—",
        ),
      );
      button.append(
        top,
        node(
          "span",
          "comparison-state",
          item.error || LABELS[item.result.state],
        ),
      );
      if (item.result) {
        const bar = node("span", "comparison-bar");
        const marker = node("b");
        marker.style.left = `${50 + Math.max(-1, Math.min(1, item.result.margin)) * 50}%`;
        bar.setAttribute("aria-hidden", "true");
        bar.append(node("i"), marker);
        button.append(bar);
      }
      button.append(
        node(
          "span",
          "comparison-select",
          dirty
            ? "Analyze again to update"
            : item.id === activeId
              ? "Current context"
              : "Inspect this context ↗",
        ),
      );
      button.addEventListener("click", () => switchContext(item.id));
      return button;
    }),
  );
}
function renderFeatures() {
  if (!current) return;
  const evidence = current.evidence.find((item) => item.id === selectedId);
  if (!evidence) return;
  const words = new Set(
    evidence.contributions.flatMap((feature) => feature.term.split(" ")),
  );
  const comment = $("highlighted-query");
  comment.replaceChildren();
  const expression = /[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu;
  let position = 0;
  for (const match of current.query.matchAll(expression)) {
    comment.append(
      document.createTextNode(current.query.slice(position, match.index)),
    );
    const text = match[0];
    comment.append(
      words.has(canonical(text))
        ? node("mark", "", text)
        : document.createTextNode(text),
    );
    position = match.index + text.length;
  }
  comment.append(document.createTextNode(current.query.slice(position)));
  $("selected-description").textContent =
    `Compared with this ${evidence.label === "allowed" ? "allowed" : "violation"} example: “${evidence.text}”`;
  $("contribution-total").textContent = `Σ = ${evidence.similarity.toFixed(3)}`;
  $("features").replaceChildren(
    ...evidence.contributions.slice(0, 12).map((feature) => {
      const chip = node(
        "span",
        `feature-chip${feature.ruleBoost > 1 ? " boosted" : ""}`,
      );
      chip.append(
        node("span", "", feature.term),
        node("b", "", feature.contribution.toFixed(3)),
      );
      chip.title = `Exact cosine contribution: ${feature.contribution.toPrecision(6)}${feature.ruleBoost > 1 ? "; shared with rule: 1.5× feature weight" : ""}`;
      return chip;
    }),
  );
  if (!evidence.contributions.length)
    $("features").append(
      node("span", "help", "No shared fitted features for this pair."),
    );
  const shown = Math.min(12, evidence.contributions.length);
  $("feature-note").textContent =
    `${shown} of ${evidence.contributions.length} shared features shown. All contributions sum to the pair’s cosine similarity. Stronger outlines mark terms that also occur in the rule; word-pair features retain local wording, not sentence meaning.`;
}
function renderEvidence() {
  if (!current) return;
  const evidence = showAll
    ? current.evidence
    : current.evidence.filter((item) => current.selectedIds.includes(item.id));
  $("evidence-list").replaceChildren(
    ...evidence.map((item) => {
      const button = node(
        "button",
        `evidence-row${item.id === selectedId ? " selected" : ""}`,
      );
      button.id = `evidence-${item.id}`;
      button.setAttribute("aria-pressed", String(item.id === selectedId));
      const left = node("span");
      const used = current.selectedIds.includes(item.id);
      const label = node(
        "span",
        `evidence-label ${item.label}`,
        `${item.label === "allowed" ? "ALLOWED" : "VIOLATION"} · ${used ? "used in margin" : "outside nearest two"}`,
      );
      left.append(label, node("span", "evidence-copy", item.text));
      button.append(
        left,
        node("span", "evidence-similarity", item.similarity.toFixed(3)),
      );
      button.setAttribute(
        "aria-label",
        `${item.label} example, cosine similarity ${item.similarity.toFixed(3)}: ${item.text}`,
      );
      button.addEventListener("click", () => {
        selectedId = item.id;
        renderEvidence();
        renderFeatures();
      });
      return button;
    }),
  );
  $("show-all").disabled =
    current.evidence.length <= current.selectedIds.length;
  $("show-all").textContent = showAll
    ? "Show nearest examples"
    : `Show all ${current.evidence.length} examples`;
}
function renderResult() {
  $("outcome").textContent = LABELS[current.state];
  $("margin").textContent = signed(current.margin);
  $("margin-marker").style.left =
    `${50 + Math.max(-1, Math.min(1, current.margin)) * 50}%`;
  $("margin-chart").setAttribute(
    "aria-label",
    `Lexical similarity margin ${signed(current.margin)}, from minus one for allowed matches to plus one for violation matches. Not a probability.`,
  );
  const qualifiers =
    current.state === "NO_EVIDENCE"
      ? "There is too little lexical overlap to suggest a boundary. "
      : current.state === "BALANCED"
        ? "Both example sets match similarly. "
        : "";
  $("score-explanation").textContent =
    `${qualifiers}Violation matches average ${current.violationSimilarity.toFixed(3)}; allowed matches average ${current.allowedSimilarity.toFixed(3)}. Their difference is a lexical margin, not a probability or moderation verdict.`;
  $("coverage").textContent = `${Math.round(current.coverage * 100)}%`;
  $("support-count").textContent = String(current.supportCount);
  $("feature-count").textContent = String(current.featureCount);
  $("result-status").textContent = "LOCAL RESULT";
  $("result-status").className = "status";
  $("export").disabled = false;
  selectedId = current.evidence[0].id;
  renderEvidence();
  renderFeatures();
}
function clearResult() {
  current = null;
  selectedId = null;
  $("outcome").textContent = "Check the context first.";
  for (const id of ["margin", "coverage", "support-count", "feature-count"])
    $(id).textContent = "—";
  $("margin-marker").style.left = "50%";
  $("margin-chart").setAttribute(
    "aria-label",
    "No valid analysis is available.",
  );
  $("score-explanation").textContent =
    "No result was produced. Resolve the input issue and analyze again.";
  $("result-status").textContent = "CHECK INPUT";
  $("result-status").className = "status stale";
  $("evidence-list").replaceChildren();
  $("features").replaceChildren();
  $("highlighted-query").textContent = $("query").value;
  $("selected-description").textContent =
    "Evidence will appear after a valid analysis.";
  $("contribution-total").textContent = "";
  $("feature-note").textContent = "";
  $("export").disabled = true;
  $("show-all").disabled = true;
}
function runAnalysis() {
  storeEditor();
  const context = contexts.find((item) => item.id === activeId);
  showAll = false;
  $("error").hidden = true;
  try {
    current = analyze(context, $("query").value);
    comparisons = compare(contexts, $("query").value);
    dirty = false;
    renderResult();
    renderComparisons();
    $("comparison-status").textContent =
      "Current comment evaluated separately against each local policy context.";
    const changes = [];
    if (current.purged)
      changes.push(
        `${current.purged} exact query match${current.purged === 1 ? "" : "es"} removed`,
      );
    if (current.duplicates)
      changes.push(
        `${current.duplicates} duplicate example${current.duplicates === 1 ? "" : "s"} removed`,
      );
    $("input-status").textContent = changes.length
      ? `Analysis complete. ${changes.join("; ")} before fitting.`
      : "Analysis complete. Vocabulary fitted on examples only; no text left this device.";
  } catch (error) {
    dirty = false;
    clearResult();
    comparisons = compare(contexts, $("query").value);
    renderComparisons();
    $("comparison-status").textContent =
      "Each context below reports its own result or input error. The selected context has no valid result.";
    $("error").textContent = error.message;
    $("error").hidden = false;
    $("input-status").textContent =
      "Analysis stopped. Edit the inputs and try again.";
  }
  updateCounts();
}
function switchContext(id) {
  storeEditor();
  activeId = id;
  loadEditor();
  runAnalysis();
}
$("policy").replaceChildren(
  ...PRESETS.map((item) => {
    const option = node("option", "", item.name);
    option.value = item.id;
    return option;
  }),
);
$("query-presets").replaceChildren(
  ...QUERIES.map((item) => {
    const button = node("button", "sample-button", item.name);
    button.addEventListener("click", () => {
      $("query").value = item.text;
      runAnalysis();
    });
    return button;
  }),
);
$("policy").addEventListener("change", () => switchContext($("policy").value));
for (const id of ["rule", "query", "violation", "allowed"])
  $(id).addEventListener("input", markDirty);
$("analyze").addEventListener("click", runAnalysis);
$("show-all").addEventListener("click", () => {
  showAll = !showAll;
  if (!showAll && !current.selectedIds.includes(selectedId))
    selectedId = current.evidence[0].id;
  renderEvidence();
  renderFeatures();
});
$("reset").addEventListener("click", () => {
  contexts = clone(PRESETS);
  activeId = contexts[0].id;
  $("query").value = QUERIES[0].text;
  loadEditor();
  runAnalysis();
  $("input-status").textContent =
    "Starter context restored. Every displayed score was recalculated locally.";
});
$("export").addEventListener("click", () => {
  if (dirty || !current) return;
  const output = exportResult(contexts, $("query").value, activeId);
  const blob = new Blob([JSON.stringify(output, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = node("a");
  link.href = url;
  link.download = "policy-lens-local-analysis.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $("input-status").textContent =
    "Downloaded this local analysis. The export includes your edited text and example contexts.";
});
$("query").value = QUERIES[0].text;
loadEditor();
runAnalysis();
