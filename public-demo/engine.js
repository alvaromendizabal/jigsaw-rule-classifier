/** Policy Lens: an authored, synthetic, lexical teaching example. No neural model. */
export const CONFIG = Object.freeze({
  version: "policy-lens-1",
  maxQuery: 1200,
  maxRule: 600,
  maxExample: 500,
  maxPerLabel: 8,
  neighbors: 2,
  ruleBoost: 1.5,
  weakSimilarity: 0.08,
  weakCoverage: 0.15,
  balancedMargin: 0.04,
});
export const PRESETS = [
  {
    id: "promotion",
    name: "Commercial promotion",
    short: "Promotion",
    rule: "Do not promote purchases, paid products or referral links. Discussion, requests for recommendations and independent reviews are allowed.",
    violation: [
      "Buy my online course with code SUNNY20 for a discount.",
      "Visit my shop and order a handmade notebook today.",
      "Subscribe to my paid tutorial series; the purchase link is in my profile.",
      "Use my referral link to sign up and earn a bonus.",
    ],
    allowed: [
      "I found a free tutorial that explains the technique clearly.",
      "The class includes a discussion of how advertising discounts work.",
      "Does anyone know a good course on drawing?",
      "I tried that notebook and wrote a review of the paper quality.",
    ],
  },
  {
    id: "attacks",
    name: "Personal attacks",
    short: "Personal attacks",
    rule: "Do not insult people or attack their character. Criticism of ideas, products and arguments is allowed. Quoting an insult to discuss the rule is allowed.",
    violation: [
      "You are clueless, and your comment is worthless.",
      "Only an idiot would write that argument.",
      "Stop posting, you useless fool.",
      "Your opinion proves that you are stupid.",
    ],
    allowed: [
      "I disagree with your argument because its evidence is incomplete.",
      "This tutorial needs a clearer explanation of the drawing technique.",
      "The phrase you are clueless is quoted here to discuss the rule, not to insult anyone.",
      "The course is overpriced, but the lesson on drawing is useful.",
    ],
  },
  {
    id: "spoilers",
    name: "Unmarked story spoilers",
    short: "Story spoilers",
    rule: "Do not reveal major story endings, character deaths or the identity of the culprit without a spoiler warning. General reviews, trailers and production discussion are allowed.",
    violation: [
      "At the end of The Lantern, Mara is revealed to be the missing pilot.",
      "The final chapter reveals that the guide stole the map.",
      "The detective partner is the culprit in the last episode.",
      "The captain dies in the closing scene.",
    ],
    allowed: [
      "I loved the lighting in The Lantern and recommend watching it.",
      "Please use a spoiler warning before discussing the ending.",
      "The trailer introduces Mara, the pilot, and an old map.",
      "I bought the course on lighting because the free tutorial was useful.",
    ],
  },
];
export const QUERIES = [
  {
    name: "A sales pitch",
    text: "I made a drawing tutorial. Buy my full course with code SUNNY20 for a discount.",
  },
  {
    name: "A quotation",
    text: "The phrase you are clueless is an example quoted to discuss the rule, not an insult.",
  },
  {
    name: "An insult",
    text: "Your comment is worthless. Stop posting, you clueless fool.",
  },
  {
    name: "A story ending",
    text: "In the last episode, the detective partner is revealed as the culprit.",
  },
];
const STOP = new Set(
  "a an the and or but of to in on at by for from with as i me my we our you your he she it its they their this that these those is are was were be been being am have has had do does did would could should can will just very also here there".split(
    " ",
  ),
);
const require = (condition, message) => {
  if (!condition) throw new Error(message);
};
export function tokens(text) {
  return (
    text
      .normalize("NFKC")
      .toLowerCase()
      .match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || []
  ).filter((t) => !STOP.has(t));
}
export function canonical(text) {
  return (
    text
      .normalize("NFKC")
      .toLowerCase()
      .match(/[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*/gu) || []
  ).join(" ");
}
export function featureCounts(text) {
  const words = tokens(text),
    counts = new Map();
  const add = (term) => counts.set(term, (counts.get(term) || 0) + 1);
  words.forEach(add);
  for (let i = 1; i < words.length; i++) add(`${words[i - 1]} ${words[i]}`);
  return counts;
}
export function fingerprint(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++)
    h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return (h >>> 0).toString(16).padStart(8, "0");
}
function checkedText(value, name, maximum) {
  require(typeof value === "string", `${name} must be text.`);
  const text = value.trim();
  require(text.length > 0, `${name} cannot be empty.`);
  require(text.length <=
    maximum, `${name} is limited to ${maximum} characters.`);
  require(canonical(text).length > 0, `${name} needs at least one word.`);
  return text;
}
export function prepareContext(input, query) {
  require(input && typeof input === "object", "A policy context is required.");
  const rule = checkedText(input.rule, "Rule", CONFIG.maxRule);
  const queryKey = canonical(query),
    seen = new Map(),
    support = [];
  let duplicates = 0,
    purged = 0;
  for (const label of ["violation", "allowed"]) {
    require(Array.isArray(input[label]) &&
      input[label].length >= 1 &&
      input[label].length <=
        CONFIG.maxPerLabel, `Provide 1–${CONFIG.maxPerLabel} ${label} examples, one per line.`);
    for (const raw of input[label]) {
      const text = checkedText(
        raw,
        `${label === "allowed" ? "Allowed" : "Violation"} example`,
        CONFIG.maxExample,
      );
      const key = canonical(text);
      if (seen.has(key)) {
        require(seen.get(key) ===
          label, "The same example has conflicting labels. Edit the examples before analyzing.");
        duplicates++;
        continue;
      }
      seen.set(key, label);
      if (key === queryKey) {
        purged++;
        continue;
      }
      support.push({ id: `${label}-${support.length}`, label, text, key });
    }
  }
  for (const label of ["violation", "allowed"])
    require(support.some(
      (row) => row.label === label,
    ), `No ${label} examples remain after exact query-overlap removal. Add a different example.`);
  return {
    id: typeof input.id === "string" ? input.id : "custom",
    name: typeof input.name === "string" ? input.name : "Custom policy",
    rule,
    support,
    duplicates,
    purged,
  };
}
export function fitVocabulary(support, rule) {
  const counts = support.map((row) => featureCounts(row.text)),
    frequency = new Map();
  for (const row of counts)
    for (const term of row.keys())
      frequency.set(term, (frequency.get(term) || 0) + 1);
  const ruleWords = new Set(tokens(rule));
  const vocabulary = [...frequency.keys()].sort().map((term) => ({
    term,
    idf: Math.log((1 + support.length) / (1 + frequency.get(term))) + 1,
    ruleBoost: term.split(" ").every((word) => ruleWords.has(word))
      ? CONFIG.ruleBoost
      : 1,
  }));
  require(vocabulary.length >
    0, "The examples contain no usable content words.");
  return vocabulary;
}
export function vectorize(text, vocabulary) {
  const counts = featureCounts(text);
  const values = vocabulary.map(({ term, idf, ruleBoost }) =>
    counts.has(term) ? (1 + Math.log(counts.get(term))) * idf * ruleBoost : 0,
  );
  const norm = Math.sqrt(values.reduce((sum, x) => sum + x * x, 0));
  return { values: norm ? values.map((x) => x / norm) : values, norm };
}
function rankEvidence(queryVector, support, vocabulary) {
  return support
    .map((row) => {
      const vector = vectorize(row.text, vocabulary);
      const contributions = vocabulary
        .flatMap((feature, i) => {
          const contribution = queryVector.values[i] * vector.values[i];
          return contribution > 0
            ? [
                {
                  term: feature.term,
                  contribution,
                  ruleBoost: feature.ruleBoost,
                },
              ]
            : [];
        })
        .sort(
          (a, b) =>
            b.contribution - a.contribution || a.term.localeCompare(b.term),
        );
      const similarity = contributions.reduce(
        (sum, feature) => sum + feature.contribution,
        0,
      );
      return {
        id: row.id,
        label: row.label,
        text: row.text,
        similarity,
        contributions,
      };
    })
    .sort((a, b) => b.similarity - a.similarity || a.id.localeCompare(b.id));
}
export function analyze(input, rawQuery) {
  const query = checkedText(rawQuery, "Comment", CONFIG.maxQuery);
  const context = prepareContext(input, query);
  const vocabulary = fitVocabulary(context.support, context.rule);
  const queryVector = vectorize(query, vocabulary);
  const evidence = rankEvidence(queryVector, context.support, vocabulary);
  const nearest = {};
  for (const label of ["violation", "allowed"])
    nearest[label] = evidence
      .filter((row) => row.label === label)
      .slice(0, CONFIG.neighbors);
  const average = (rows) =>
    rows.reduce((sum, row) => sum + row.similarity, 0) / rows.length;
  const violationSimilarity = average(nearest.violation),
    allowedSimilarity = average(nearest.allowed);
  const margin = violationSimilarity - allowedSimilarity;
  const queryWords = [...new Set(tokens(query))],
    known = new Set(vocabulary.map((v) => v.term));
  const matchedWords = queryWords.filter((word) => known.has(word));
  const coverage = queryWords.length
    ? matchedWords.length / queryWords.length
    : 0;
  const strongest = evidence[0].similarity;
  const state =
    strongest < CONFIG.weakSimilarity || coverage < CONFIG.weakCoverage
      ? "NO_EVIDENCE"
      : Math.abs(margin) < CONFIG.balancedMargin
        ? "BALANCED"
        : margin > 0
          ? "VIOLATION_MATCH"
          : "ALLOWED_MATCH";
  return {
    scope: "PUBLIC_LEXICAL_DEMONSTRATION",
    engine: CONFIG.version,
    contextId: context.id,
    contextName: context.name,
    query,
    rule: context.rule,
    state,
    margin,
    violationSimilarity,
    allowedSimilarity,
    coverage,
    matchedWords,
    queryWords,
    evidence,
    selectedIds: [...nearest.violation, ...nearest.allowed].map(
      (row) => row.id,
    ),
    supportCount: context.support.length,
    featureCount: vocabulary.length,
    purged: context.purged,
    duplicates: context.duplicates,
    vocabularyFingerprint: fingerprint(JSON.stringify(vocabulary)),
    interpretation:
      "Signed lexical similarity margin, not a probability, calibrated confidence, moderation verdict or neural-model result.",
    limitations: [
      "Surface words cannot reliably resolve intent, sarcasm, quotations or negation.",
      "Examples define the labels; rule wording only weights shared terms.",
      "No validation score or accuracy is estimated from these editable examples.",
    ],
  };
}
export function compare(contexts, query) {
  return contexts.map((context) => {
    try {
      return {
        id: context.id,
        name: context.name,
        result: analyze(context, query),
        error: null,
      };
    } catch (error) {
      return {
        id: context.id,
        name: context.name,
        result: null,
        error: error.message,
      };
    }
  });
}
export function exportResult(contexts, query, activeId) {
  return {
    schema: 1,
    starterScope: "SYNTHETIC_ONLY",
    inputScope: "EDITABLE_LOCAL_TEXT",
    engine: CONFIG,
    purpose:
      "Authored fictional policy examples and editable local text. No competition data, private model, neural inference or accuracy claim.",
    activeContext: activeId,
    query,
    contexts: JSON.parse(JSON.stringify(contexts)),
    comparisons: compare(contexts, query),
  };
}
