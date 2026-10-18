"""Target-free support-adaptation plans and resumable decision-token learning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from scripts.competition_features import prepare_prompts, support_pairs


def build_study(frame: pd.DataFrame) -> dict:
    """Mimic a new rule with legitimate known supports and genuinely novel queries.

    Query eligibility depends on text only. No query label is exported. Every
    query body is excluded from all adaptation sources, including other rules.
    """
    validate_frame(frame, train=True)
    available = frame.drop(columns="rule_violation")
    known_text = {normalize(text) for name in EXAMPLES for text in frame[name]}
    folds = []
    for rule in sorted(frame.rule.unique()):
        eligible = (frame.rule == rule) & ~frame.body.map(normalize).isin(known_text)
        query = frame.loc[eligible].sort_values("row_id")
        if query.empty:
            raise ValueError("No novel-comment evaluation cohort")
        training, audit = support_pairs(
            frame.loc[frame.rule != rule], available, forbidden=query.body
        )
        rows = training[["body", "rule", "rule_violation"]].to_dict("records")
        for row in rows:
            row["repeat"] = 2 if normalize(row["rule"]) == normalize(rule) else 1
        folds.append(
            {
                "rule": rule,
                "training": rows,
                "queries": query[["row_id", "body", "rule"]].to_dict("records"),
                "audit": {
                    **audit,
                    "candidate_query_rows": int((frame.rule == rule).sum()),
                    "novel_query_rows": len(query),
                    "known_support_overlap_excluded": int((frame.rule == rule).sum()) - len(query),
                    "new_rule_training_pairs": sum(row["repeat"] == 2 for row in rows),
                    "epoch_occurrences": sum(row["repeat"] for row in rows),
                },
            }
        )
    plan = {"schema": 1, "protocol": "new_rule_supplied_support_adaptation", "folds": folds}
    validate_study(plan)
    return plan


def validate_study(plan: dict) -> None:
    seen_queries = set()
    for fold in plan["folds"]:
        query_text = set()
        for row in fold["queries"]:
            if set(row) != {"row_id", "body", "rule"}:
                raise ValueError("Query schema must exclude targets")
            if row["row_id"] in seen_queries or row["rule"] != fold["rule"]:
                raise ValueError("Duplicate query ID or inconsistent policy")
            seen_queries.add(row["row_id"])
            query_text.add(normalize(row["body"]))
        keys = set()
        for row in fold["training"]:
            if set(row) != {"body", "rule", "rule_violation", "repeat"}:
                raise ValueError("Unexpected adaptation training schema")
            if row["rule_violation"] not in (0, 1) or row["repeat"] not in (1, 2):
                raise ValueError("Invalid adaptation label or repetition")
            key = (normalize(row["rule"]), normalize(row["body"]))
            if key in keys or key[1] in query_text:
                raise ValueError("Adaptation contains duplicate pairs or evaluation text")
            keys.add(key)


def decision_prompts(rows: list[dict], tokenizer, spec: dict) -> list[str]:
    """Reuse the exact frozen rule-only template for a matched comparison."""
    frame = pd.DataFrame(rows)[["body", "rule"]].assign(
        row_id=np.arange(len(rows)), subreddit="unused", **{name: "unused" for name in EXAMPLES}
    )
    texts, _ = prepare_prompts(frame, tokenizer, spec)
    return texts[::3]


def add_adapter(model, spec: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec["rank"],
            lora_alpha=spec["alpha"],
            lora_dropout=spec["dropout"],
            target_modules=spec["target_modules"],
            bias="none",
        ),
    )
