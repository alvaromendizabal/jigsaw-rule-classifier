"""Inspectable behavioral cues; supplied-support contrasts are inner-group cross-fitted.

Heuristics are measurements, never labels. No URL is fetched. No query target is read
by feature construction. Single quotes and nuanced quotation/negation remain limitations.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from jigsaw_rules.data import normalize

CUES = {
    "request": (
        "\\b(?:can|could|should|would)\\s+(?:i|we|you)\\b|\\bhow\\s+(?:do|can|s"
        "hould)\\b|\\b(?:any advice|need help)\\b"
    ),
    "offer": (
        "\\bi\\s+can\\s+(?:help|advise|offer)\\b|\\b(?:happy to help|contact me|message me)\\b"
    ),
    "directive": r"\byou\s+(?:should|must|need to|ought to)\b|\bi\s+(?:recommend|suggest)\b",
    "conditional": r"\b(?:if|unless|provided that|in case)\b",
    "hedge": r"\b(?:might|may|could|perhaps|probably|possibly|it depends)\b",
    "negation": r"\b(?:not|no|never|cannot|can't|don't|doesn't|isn't|won't)\b",
    "disclaimer": (
        "\\bnot\\s+(?:a|your)\\s+(?:lawyer|attorney|doctor)\\b|\\bnot\\s+(?:lega"
        "l|medical|financial)\\s+advice\\b|\\bfor informational purposes\\b"
    ),
    "anecdote": r"\bi\s+(?:was|had|did|experienced)\b|\bin my experience\b",
    "attribution": r"\b(?:they said|he said|she said|was told|according to|wrote that|quoted)\b",
    "meta": r"\b(?:this rule|moderators?|subreddit|this post|community guidelines)\b",
    "promotion": r"\b(?:sale|discount|coupon|promo(?:tion)?|referral|subscribe|special offer)\b",
    "self_promotion": (
        r"\b(?:my|our)\s+(?:shop|store|website|channel|business|product)\b"
        r"|\bwe sell\b"
    ),
    "transaction": r"\b(?:purchase|payment|price|cost|shipping|fee|buy|order)\b",
    "direct_contact": r"\b(?:dm|pm|email|message me|contact me|reach me)\b",
    "offplatform": r"\b(?:whatsapp|telegram|discord)\b",
    "legal_topic": (
        "\\b(?:court|lawyer|attorney|legal|statute|lawsuit|lease|landlord|contract|sue)\\b"
    ),
    "legal_action": (
        "\\b(?:sue|file a claim|file a lawsuit|appeal|press charges|hire an"
        " attorney|sign the contract)\\b"
    ),
    "commercial_action": (
        "\\b(?:buy|subscribe|order|purchase|sign up|use my code|use our code|click here)\\b"
    ),
    "question": r"\?",
    "first_person": r"\b(?:i|me|my|we|our)\b",
    "second_person": r"\b(?:you|your|yours)\b",
    "urgency": r"\b(?:today only|limited time|act now|hurry|last chance|immediately)\b",
    "resource": r"\b(?:source|reference|citation|documentation|study|information)\b",
    "url": r"https?://[^\s<>]+|www\.[^\s<>]+",
    "money": r"[$€£]\s*\d|\b\d+(?:\.\d+)?\s*(?:usd|dollars?|euros?)\b",
    "refusal": r"\b(?:cannot|can't|won't|do not)\s+(?:give|offer|provide)\b",
    "jurisdiction": r"\b(?:jurisdiction|state law|local law|federal law|in your state)\b",
    "exception": r"\b(?:except|unless|however|although|but)\b",
}
COMPILED = {k: re.compile(v, re.I) for k, v in CUES.items()}
FAMILIES = ("behavior", "scope", "rule_alignment", "support_contrast")
RULE_MAP = {
    "request": r"\brequest",
    "offer": r"\boffer",
    "legal_topic": r"\blegal",
    "legal_action": r"\blegal",
    "promotion": r"advertis|promot|spam",
    "commercial_action": r"advertis|promot|spam",
    "self_promotion": r"advertis|promot|spam",
    "direct_contact": r"solicit|spam|advertis",
    "url": r"links?|spam|advertis",
    "referral": r"referral",
    "exception": r"except|unless",
    "quote": r"quot|cit",
}


def valid_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Feature inputs must be nonempty strings")
    return unicodedata.normalize("NFKC", value).replace("’", "'")


def split_scope(text: str) -> tuple[str, str, str, dict]:
    text = valid_text(text)
    code_mask = np.zeros(len(text), dtype=bool)
    quote_mask = np.zeros(len(text), dtype=bool)
    for pattern in (r"```[\s\S]*?```", r"`[^`\n]+`"):
        for match in re.finditer(pattern, text):
            code_mask[match.start() : match.end()] = True
    for pattern in (r'"[^"\n]+"', r"“[^”\n]+”", r"(?m)^\s*>[^\n]*"):
        for match in re.finditer(pattern, text):
            quote_mask[match.start() : match.end()] = True
    quote_mask &= ~code_mask
    authored_mask = ~(quote_mask | code_mask)

    def project(mask):
        return "".join(ch if keep else " " for ch, keep in zip(text, mask, strict=True))

    meta = {
        "quoted_fraction": float(quote_mask.mean()),
        "code_fraction": float(code_mask.mean()),
        "authored_fraction": float(authored_mask.mean()),
        "unmatched_double_quote": float(text.count('"') % 2),
    }
    return project(authored_mask), project(quote_mask), project(code_mask), meta


def cue_counts(text: str) -> np.ndarray:
    return np.asarray([min(len(p.findall(text)), 20) for p in COMPILED.values()], dtype=float)


@lru_cache(maxsize=20000)
def text_features(body: str, rule: str) -> tuple[tuple, tuple]:
    text, rule = valid_text(body), valid_text(rule)
    authored, quoted, code, meta = split_scope(text)
    full, own, cited, coded = [cue_counts(x) for x in (text, authored, quoted, code)]
    out = {}
    words = max(1, len(re.findall(r"\b\w+\b", text)))
    for j, name in enumerate(CUES):
        out[f"behavior/{name}_present"] = float(full[j] > 0)
        out[f"behavior/{name}_log_count"] = float(np.log1p(full[j]))
        out[f"scope/{name}_authored"] = float(np.log1p(own[j]))
        out[f"scope/{name}_quoted"] = float(np.log1p(cited[j]))
        out[f"scope/{name}_code"] = float(np.log1p(coded[j]))
        out[f"scope/{name}_authored_minus_quoted"] = float(np.log1p(own[j]) - np.log1p(cited[j]))
    out.update({f"scope/{key}": value for key, value in meta.items()})
    out["behavior/log_words"] = float(np.log1p(words))
    out["behavior/log_chars"] = float(np.log1p(len(text)))
    # Sentence-local associations are intentionally not a semantic parser.
    sentences = re.split(r"[.!?;\n]+", authored)
    negated = {k: 0 for k in ("request", "offer", "directive", "commercial_action", "legal_action")}
    for sentence in sentences:
        for name in negated:
            for match in COMPILED[name].finditer(sentence):
                preceding = " ".join(sentence[: match.start()].split()[-4:])
                negated[name] += bool(COMPILED["negation"].search(preceding))
    for name, count in negated.items():
        out[f"scope/{name}_preceded_by_negation"] = float(np.log1p(count))
    index = {name: i for i, name in enumerate(CUES)}
    for rule_cue, pattern in RULE_MAP.items():
        active = bool(re.search(pattern, rule, re.I))
        name = "promotion" if rule_cue == "referral" else rule_cue
        values = (
            (meta["quoted_fraction"], meta["code_fraction"])
            if name == "quote"
            else (np.log1p(own[index[name]]), np.log1p(cited[index[name]]))
        )
        out[f"rule_alignment/{rule_cue}_authored_match"] = float(active * values[0])
        out[f"rule_alignment/{rule_cue}_quoted_match"] = float(active * values[1])
    for action in ("request", "offer", "directive", "refusal", "anecdote", "disclaimer"):
        for topic, pattern in (("legal_topic", r"legal"), ("promotion", r"advertis|spam|promot")):
            relevant = bool(re.search(pattern, rule, re.I))
            together = sum(
                bool(COMPILED[action].search(s) and COMPILED[topic].search(s)) for s in sentences
            )
            out[f"rule_alignment/{action}_with_{topic}"] = float(relevant * np.log1p(together))
    return tuple(out), tuple(out.values())


def basic_features(rows: pd.DataFrame) -> pd.DataFrame:
    if not {"body", "rule"}.issubset(rows):
        raise ValueError("Feature inputs need body and rule")
    records = [
        text_features(body, rule)
        for body, rule in rows[["body", "rule"]].itertuples(index=False, name=None)
    ]
    if not records:
        raise ValueError("Empty feature frame")
    result = pd.DataFrame([r[1] for r in records], columns=records[0][0])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite behavioral features")
    return result


def behavior_vectors(rows: pd.DataFrame) -> np.ndarray:
    """Fixed cue space only: no fitted vocabulary or target information."""
    out = []
    for body in rows.body:
        own, quoted, _, _ = split_scope(body)
        out.append(np.r_[np.log1p(cue_counts(own)), np.log1p(cue_counts(quoted))])
    return np.asarray(out, dtype=float)


def support_features(query: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if not {"body", "rule", "rule_violation"}.issubset(reference):
        raise ValueError("Reference requires explicitly supplied labels")
    if not reference.rule_violation.isin([0, 1]).all():
        raise ValueError("Reference labels must be binary")
    qbody, rbody = query.body.map(normalize), reference.body.map(normalize)
    if set(qbody) & set(rbody):
        raise ValueError("Query text leaked into reference pool")
    qv, rv = behavior_vectors(query), behavior_vectors(reference)
    qu = qv / np.maximum(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12)
    ru = rv / np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-12)
    keys = [f"{scope}_{name}" for scope in ("authored", "quoted") for name in CUES]
    output = np.zeros((len(query), 9 + len(keys)), dtype=float)
    qr, rr = query.rule.map(normalize).to_numpy(), reference.rule.map(normalize).to_numpy()
    y = reference.rule_violation.to_numpy(dtype=int)
    for rule in np.unique(qr):
        qi = np.flatnonzero(qr == rule)
        zero = np.flatnonzero((rr == rule) & (y == 0))
        one = np.flatnonzero((rr == rule) & (y == 1))
        if not len(one) or not len(zero):
            raise ValueError("Same-rule reference must contain both explicit classes")
        pos, neg = qu[qi] @ ru[one].T, qu[qi] @ ru[zero].T
        pmax, nmax = pos.max(axis=1), neg.max(axis=1)
        pmean, nmean = pos.mean(axis=1), neg.mean(axis=1)
        ptop = np.sort(pos, axis=1)[:, -min(3, len(one)) :].mean(axis=1)
        ntop = np.sort(neg, axis=1)[:, -min(3, len(zero)) :].mean(axis=1)
        output[qi, :9] = np.column_stack(
            (
                pmax,
                nmax,
                pmax - nmax,
                pmean,
                nmean,
                pmean - nmean,
                ptop - ntop,
                pos.std(axis=1),
                neg.std(axis=1),
            )
        )
        # Positive values indicate closer cue rates to violating references, not a label.
        pm, nm = rv[one].mean(axis=0), rv[zero].mean(axis=0)
        output[qi, 9:] = np.abs(qv[qi] - nm) - np.abs(qv[qi] - pm)
    names = [
        "positive_max",
        "negative_max",
        "max_margin",
        "positive_mean",
        "negative_mean",
        "mean_margin",
        "top3_margin",
        "positive_spread",
        "negative_spread",
    ]
    names += [f"relative_distance_{k}" for k in keys]
    return pd.DataFrame(output, columns=[f"support_contrast/{k}" for k in names])


def cross_fitted_support(training: pd.DataFrame, query: pd.DataFrame, seed: int = 20260911):
    groups = training.body.map(normalize).to_numpy()
    strata = training.rule.map(normalize) + "::" + training.rule_violation.astype(str)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    out = None
    seen = np.zeros(len(training), dtype=int)
    audits = []
    for tr, va in splitter.split(training, strata, groups):
        ref, held = training.iloc[tr], training.iloc[va][["body", "rule"]]
        block = support_features(held, ref)
        if out is None:
            out = pd.DataFrame(np.zeros((len(training), block.shape[1])), columns=block.columns)
        out.iloc[va] = block.to_numpy()
        seen[va] += 1
        audits.append({"reference_rows": len(tr), "held_rows": len(va), "body_overlap": 0})
    if not (seen == 1).all():
        raise ValueError("Inner cross-fitting is incomplete")
    return out, support_features(query[["body", "rule"]], training), audits


def screen_family(train_x: np.ndarray, y: np.ndarray, names: list[str], budget: int):
    "Outer-training-only screening; deterministic standardized mean differences."
    x, y = np.asarray(train_x, float), np.asarray(y)
    if x.ndim != 2 or x.shape[1] != len(names) or len(x) != len(y):
        raise ValueError("Screen alignment differs")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Invalid screen inputs")
    if budget < 1:
        raise ValueError("Screen budget must be positive")
    variable = np.ptp(x, axis=0) > 1e-12
    effect = np.abs(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)) / (x.std(axis=0) + 1e-8)
    order = sorted(np.flatnonzero(variable), key=lambda i: (-effect[i], names[i]))
    selected, signatures = [], set()
    for i in order:
        signature = np.ascontiguousarray(x[:, i]).tobytes()
        if signature in signatures:
            continue
        selected.append(i)
        signatures.add(signature)
        if len(selected) == budget:
            break
    return np.asarray(selected, dtype=int), {
        "candidates": len(names),
        "constant": int((~variable).sum()),
        "retained": len(selected),
        "names": [names[i] for i in selected],
    }
