"""Local action/argument relationships, not moderation labels or a semantic parser.

All inputs are observable comment text. No URL is requested, no label is inspected,
and no unobserved thread, speaker identity or history is manufactured.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from scripts.behavioral_features import valid_text

FAMILIES = ("act_roles", "link_intent", "action_scope")
LEGAL = "\\b(?:sue|suing|lawyer|attorney|lawsuit|legal|court|claim|litigation|police|appeal)\\b"
LEGAL_RE = re.compile(LEGAL, re.I)
URL = re.compile(r"(?:https?://|www\.)[^\s<>\"`]+", re.I)
TOKENS = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+", re.I)
DIRECTIVE = re.compile(
    ("\\byou\\s+(?:should|must|need to|ought to)\\b|\\bi\\s+(?:recommend|suggest)\\b"), re.I
)
REQUEST = re.compile(
    ("\\b(?:can|could|should|may)\\s+(?:i|we)\\b|\\bhow\\s+(?:do|can|should)\\s+(?:i|we)\\b"), re.I
)
OTHER_REQUEST = re.compile(r"\b(?:can|could|would)\s+(?:you|someone|anyone)\b", re.I)
OFFER = re.compile(r"\bi\s+can\s+(?:help|advise|assist|represent)\b|\bhappy to help\b", re.I)
EXPERIENCE = re.compile(r"\bi\s+(?:was|had|did|went|sued|hired)\b|\bin my experience\b", re.I)
REPORTED = re.compile(
    ("\\b(?:he|she|they)\\s+(?:said|told|wrote)\\b|\\b(?:according to|was told)\\b"), re.I
)
CONDITION = re.compile(r"\b(?:if|unless|provided that)\b", re.I)
RESOURCE = re.compile(
    ("\\b(?:read|see|source|reference|documentation|article|statute|report)\\b"), re.I
)
CTA = re.compile(
    ("\\b(?:buy|purchase|order|subscribe|sign up|click|visit|check out|use my code)\\b"), re.I
)
SELF = re.compile(r"\b(?:my|our)\s+(?:shop|store|site|website|channel|product|business)\b", re.I)
CONTACT = re.compile(r"\b(?:dm|pm|email|message me|contact me|reach me)\b", re.I)
MONEY = re.compile(
    ("\\b(?:price|payment|fee|discount|sale|coupon|promo|commission|shipping)\\b|[$€£]\\s*\\d"),
    re.I,
)
AFFILIATE = re.compile(r"\b(?:affiliate|referral|commission|sponsored)\b", re.I)
DISCLAIMER = re.compile(r"\bnot\s+(?:a|your)\s+(?:lawyer|attorney)\b|\bnot legal advice\b", re.I)
NEGATIVE = {"not", "never", "cannot", "can't", "don't", "doesn't", "won't", "shouldn't"}
ROLES = (
    "self_request_legal",
    "other_request_legal",
    "directive_legal",
    "offer_legal",
    "experience_legal",
    "reported_legal",
    "question_legal",
    "conditional_directive_legal",
    "resource_legal",
    "legal_without_guidance",
    "directive_without_legal",
    "request_without_legal",
)
LINKS = (
    "cta_url",
    "owned_business_url",
    "resource_url",
    "contact_url",
    "price_url",
    "affiliate_url",
    "referral_parameter",
    "cta_price",
    "contact_price",
    "owned_business_cta",
    "quoted_cta_url",
    "url_without_cta",
)
SCOPES = (
    "directive_legal_negated",
    "self_request_legal_negated",
    "cta_negated",
    "directive_legal_positive",
    "cta_positive",
    "quoted_directive_legal",
    "quoted_self_request_legal",
    "quoted_cta",
    "disclaimer_then_directive_legal",
    "disclaimer_only_legal",
    "reported_directive_legal",
    "conditional_cta",
)
CATALOG = {
    family: [f"{family}/{key}_{stat}" for key in keys for stat in ("present", "log_count")]
    for family, keys in zip(FAMILIES, (ROLES, LINKS, SCOPES), strict=True)
}


def channels(text: str) -> tuple[list[str], list[str]]:
    """Remove code, separate quoted material, never bind across a removed span."""
    text = valid_text(text)
    code = [
        (m.start(), m.end())
        for p in (r"```[\s\S]*?```", r"`[^`\n]+`")
        for m in re.finditer(p, text)
    ]
    quoted = [
        (m.start(), m.end())
        for p in (r'"[^"\n]+"', r"“[^”\n]+”", r"(?m)^\s*>[^\n]*")
        for m in re.finditer(p, text)
    ]
    mask = np.zeros(len(text), dtype=np.int8)
    for a, b in quoted:
        mask[a:b] = 1
    for a, b in code:
        mask[a:b] = 2
    groups = {0: [], 1: []}
    for mode in (0, 1):
        active = mask == mode
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
        groups[mode] = [text[a:b] for a, b in zip(starts, ends, strict=True)]
    return groups[0], groups[1]


def clauses(texts: list[str]) -> list[str]:
    """Protect URL punctuation; split sentences and explicit contrast boundaries."""
    out = []
    for text in texts:
        protected = list(text)
        for m in URL.finditer(text):
            for i in range(m.start(), m.start() + len(m.group().rstrip(".!;,):"))):
                if protected[i] in ".?!;":
                    protected[i] = "\ue000" if protected[i] == "." else protected[i]
        segments = re.split(
            ("[.!;\\n]+|(?<=\\?)\\s+|\\b(?:but|however|although)\\b"),
            "".join(protected),
            flags=re.I,
        )
        out.extend(s.replace("\ue000", ".") for s in segments if s.strip())
    return out


def close(a: re.Match, b: re.Match, text: str, window: int = 12) -> bool:
    left, right = sorted((a, b), key=lambda m: m.start())
    return len(TOKENS.findall(text[left.end() : right.start()])) <= window


def paired(first: re.Pattern, second: re.Pattern, text: str) -> list[tuple]:
    return [
        (a, b) for a in first.finditer(text) for b in second.finditer(text) if close(a, b, text)
    ]


def negated(predicate: re.Match, action: re.Match, text: str) -> bool:
    """Only approximate local scope between predicate/action or right before predicate."""
    prior = TOKENS.findall(text[: predicate.start()].lower())[-3:]
    inside = TOKENS.findall(text[predicate.start() : action.end()].lower())
    # Avoid a preceding identity disclaimer negating a later independent directive.
    return any(t in NEGATIVE for t in inside) or any(t in NEGATIVE for t in prior[-2:])


@lru_cache(maxsize=12000)
def relation_row(body: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
    own, quote = channels(body)
    original = clauses(own)
    cited = clauses(quote)
    roles = dict.fromkeys(ROLES, 0)
    links = dict.fromkeys(LINKS, 0)
    scope = dict.fromkeys(SCOPES, 0)
    previous_disclaimer = False
    for text in original:
        legal = bool(LEGAL_RE.search(text))
        rd = paired(DIRECTIVE, LEGAL_RE, text)
        rq = paired(REQUEST, LEGAL_RE, text)
        roles["self_request_legal"] += len(rq)
        roles["other_request_legal"] += len(paired(OTHER_REQUEST, LEGAL_RE, text))
        roles["directive_legal"] += len(rd)
        roles["offer_legal"] += len(paired(OFFER, LEGAL_RE, text))
        roles["experience_legal"] += len(paired(EXPERIENCE, LEGAL_RE, text))
        roles["reported_legal"] += len(paired(REPORTED, LEGAL_RE, text))
        roles["question_legal"] += legal and "?" in text
        roles["conditional_directive_legal"] += bool(rd and CONDITION.search(text))
        roles["resource_legal"] += len(paired(RESOURCE, LEGAL_RE, text))
        roles["legal_without_guidance"] += legal and not (rd or rq or OFFER.search(text))
        roles["directive_without_legal"] += bool(DIRECTIVE.search(text)) and not legal
        roles["request_without_legal"] += bool(REQUEST.search(text)) and not legal
        scope["directive_legal_negated"] += sum(negated(a, b, text) for a, b in rd)
        scope["directive_legal_positive"] += sum(not negated(a, b, text) for a, b in rd)
        scope["self_request_legal_negated"] += sum(negated(a, b, text) for a, b in rq)
        scope["reported_directive_legal"] += bool(rd and REPORTED.search(text))
        for m in CTA.finditer(text):
            n = any(t in NEGATIVE for t in TOKENS.findall(text[: m.start()].lower())[-3:])
            scope["cta_negated"] += n
            scope["cta_positive"] += not n
            scope["conditional_cta"] += bool(CONDITION.search(text))
        dis = bool(DISCLAIMER.search(text))
        scope["disclaimer_then_directive_legal"] += bool(rd and (dis or previous_disclaimer))
        scope["disclaimer_only_legal"] += dis and not rd
        previous_disclaimer = dis
        for pattern, key in (
            (CTA, "cta_url"),
            (SELF, "owned_business_url"),
            (RESOURCE, "resource_url"),
            (CONTACT, "contact_url"),
            (MONEY, "price_url"),
            (AFFILIATE, "affiliate_url"),
        ):
            links[key] += len(paired(pattern, URL, text))
        for pattern, other, key in (
            (CTA, MONEY, "cta_price"),
            (CONTACT, MONEY, "contact_price"),
            (SELF, CTA, "owned_business_cta"),
        ):
            links[key] += len(paired(pattern, other, text))
        urls = list(URL.finditer(text))
        links["referral_parameter"] += sum(
            bool(re.search(("[?&](?:ref|referral|aff|affiliate|coupon|promo)="), m.group(), re.I))
            for m in urls
        )
        links["url_without_cta"] += bool(urls) and not CTA.search(text)
    for text in cited:
        scope["quoted_directive_legal"] += len(paired(DIRECTIVE, LEGAL_RE, text))
        scope["quoted_self_request_legal"] += len(paired(REQUEST, LEGAL_RE, text))
        scope["quoted_cta"] += len(list(CTA.finditer(text)))
        links["quoted_cta_url"] += len(paired(CTA, URL, text))
    output = {}
    for family, values in zip(FAMILIES, (roles, links, scope), strict=True):
        for key, value in values.items():
            count = min(int(value), 20)
            output[f"{family}/{key}_present"] = float(count > 0)
            output[f"{family}/{key}_log_count"] = float(np.log1p(count))
    return tuple(output), tuple(output.values())


def relational_features(rows: pd.DataFrame) -> pd.DataFrame:
    if "body" not in rows or rows.empty:
        raise ValueError("nonempty body column required")
    pairs = [relation_row(body) for body in rows.body]
    out = pd.DataFrame([values for _, values in pairs], columns=pairs[0][0])
    if out.shape[1] != 72 or not np.isfinite(out.to_numpy()).all():
        raise ValueError("relational feature schema or finiteness failure")
    return out
