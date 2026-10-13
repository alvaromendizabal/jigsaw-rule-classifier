/** The expected label never enters analyze(): the shipped engine sees only text and policy. */
export const sources = ["public-demo/engine.js"];
export const groupKey = "policy";
export async function loadEngine(read) {
  return import(`data:text/javascript;base64,${Buffer.from(read(sources[0])).toString("base64")}`);
}
export function inspectCase(row, engine, require) {
  require(Object.keys(row).sort().join(",") === "expected,id,policy,query,slice", "Unexpected challenge fields");
  require(engine.PRESETS.some(policy => policy.id === row.policy), "Unknown policy");
  require(["allowed", "violation", "abstain"].includes(row.expected), "Invalid expected decision");
  require(row.query.length <= engine.CONFIG.maxQuery, "Oversized policy query");
}
export function runCase(row, engine) {
  const result = engine.analyze(engine.PRESETS.find(policy => policy.id === row.policy), row.query);
  const decision = result.state === "VIOLATION_MATCH" ? "violation" : result.state === "ALLOWED_MATCH" ? "allowed" : "abstain";
  return {
    id: row.id, slice: row.slice, policy: row.policy, query: row.query, expected: row.expected,
    decision, state: result.state, correct: decision === row.expected,
    margin: Number(result.margin.toFixed(12)), coverage: Number(result.coverage.toFixed(12)),
    query_support_rows_purged: result.purged,
  };
}
const rate = (n, d) => d ? Number((n / d).toFixed(12)) : null;
export function summarize(rows) {
  const labeled = rows.filter(row => row.expected !== "abstain");
  const decided = labeled.filter(row => row.decision !== "abstain");
  const unsupported = rows.filter(row => row.expected === "abstain");
  return {
    cases: rows.length,
    correct: rows.filter(row => row.correct).length,
    failures: rows.filter(row => !row.correct).length,
    agreement: rate(rows.filter(row => row.correct).length, rows.length),
    labeled_cases: labeled.length,
    decided_labeled_cases: decided.length,
    decision_coverage: rate(decided.length, labeled.length),
    selective_agreement: rate(decided.filter(row => row.correct).length, decided.length),
    false_violations: rows.filter(row => row.expected === "allowed" && row.decision === "violation").length,
    false_allowances: rows.filter(row => row.expected === "violation" && row.decision === "allowed").length,
    unsupported_cases: unsupported.length,
    unsupported_abstentions: unsupported.filter(row => row.decision === "abstain").length,
  };
}
