from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.top_solution_integration import (
    adaptive_bar_height,
    generalized_cross_entropy_hard,
    grouped_logit_margin,
    most_uncertain,
    prompt_payload,
    rank_within_group,
    resolve_supervision,
    soft_binary_cross_entropy_from_margin,
    verbalizer_words,
)


def sample_rows():
    return pd.DataFrame(
        [
            {"body": "same", "rule": "Rule A", "subreddit": "x", "rule_violation": 1},
            {"body": "same", "rule": "Rule A", "subreddit": "y", "rule_violation": 1},
            {"body": "same", "rule": "Rule A", "subreddit": "z", "rule_violation": 0},
            {"body": "tie", "rule": "Rule A", "subreddit": "x", "rule_violation": 1},
            {"body": "tie", "rule": "Rule A", "subreddit": "y", "rule_violation": 0},
            {"body": "clean no", "rule": "Rule A", "subreddit": "q", "rule_violation": 0},
            {"body": "clean yes", "rule": "Rule B", "subreddit": "q", "rule_violation": 1},
        ]
    )


def test_soft_resolution_keeps_empirical_probability_and_ignores_subreddit():
    out, audit = resolve_supervision(sample_rows(), conflict_mode="soft")
    row = out[(out.normalized_rule == "rule a") & (out.normalized_body == "same")].iloc[0]
    assert row.target_probability == pytest.approx(2 / 3)
    assert row.occurrence_count == 3
    assert audit["subreddit_used"] is False


def test_majority_resolution_uses_strict_majority_and_drops_tie():
    out, _ = resolve_supervision(sample_rows(), conflict_mode="majority")
    assert "tie" not in set(out.normalized_body)
    row = out[out.normalized_body == "same"].iloc[0]
    assert row.hard_label == 1


def test_drop_resolution_matches_current_conflict_policy():
    out, audit = resolve_supervision(sample_rows(), conflict_mode="drop")
    assert "same" not in set(out.normalized_body)
    assert "tie" not in set(out.normalized_body)
    assert audit["conflicting_pairs"] == 2


def test_forbidden_body_is_removed_independently_of_rule():
    out, audit = resolve_supervision(
        sample_rows(), conflict_mode="soft", forbidden_bodies=["CLEAN YES"]
    )
    assert "clean yes" not in set(out.normalized_body)
    assert audit["forbidden_occurrences_removed"] == 1


@pytest.mark.parametrize(
    "style",
    ["current", "concise_binary", "numeric_structured", "compliance"],
)
def test_prompt_families_exclude_subreddit(style):
    payload = prompt_payload(body="comment", rule="rule", style=style)
    rendered = " ".join(payload.values()).lower()
    assert "subreddit" not in rendered
    assert "comment" in rendered and "rule" in rendered


def test_expanded_verbalizers_add_single_letter_variants():
    base = verbalizer_words(expanded=False)
    expanded = verbalizer_words(expanded=True)
    assert "Y" not in base["positive"] and "N" not in base["negative"]
    assert "Y" in expanded["positive"] and "N" in expanded["negative"]


def test_grouped_margin_aggregates_multiple_decision_tokens():
    logits = np.zeros((2, 8), dtype=float)
    logits[0, [1, 2]] = [0.0, 1.0]
    logits[0, [5, 6]] = [2.0, 1.0]
    logits[1, [1, 2]] = [2.0, 1.0]
    logits[1, [5, 6]] = [0.0, 1.0]
    margin = grouped_logit_margin(logits, negative_ids=[1, 2], positive_ids=[5, 6])
    assert margin[0] > 0
    assert margin[1] < 0


def test_soft_ce_is_finite_and_prefers_correct_margin():
    good = soft_binary_cross_entropy_from_margin(np.array([3.0]), np.array([1.0]))
    bad = soft_binary_cross_entropy_from_margin(np.array([-3.0]), np.array([1.0]))
    assert np.isfinite(good).all()
    assert good[0] < bad[0]


def test_gce_hard_is_finite():
    loss = generalized_cross_entropy_hard(np.array([0.9, 0.1, 0.55]), np.array([1, 0, 1]), q=0.955)
    assert np.isfinite(loss).all()
    assert (loss >= 0).all()


def test_rank_within_rule_never_cross_contaminates_rules():
    result = rank_within_group([10, 20, 100, 200], ["a", "a", "b", "b"])
    np.testing.assert_allclose(result, [0.25, 0.75, 0.25, 0.75])


def test_uncertainty_selection_is_group_balanced_and_deterministic():
    probabilities = [0.01, 0.49, 0.51, 0.99, 0.48, 0.52]
    groups = ["a", "a", "a", "b", "b", "b"]
    first = most_uncertain(probabilities, count=4, groups=groups)
    second = most_uncertain(probabilities, count=4, groups=groups)
    np.testing.assert_array_equal(first, second)
    assert sum(np.asarray(groups)[first] == "a") == 2
    assert sum(np.asarray(groups)[first] == "b") == 2


def test_adaptive_plot_height_is_bounded_and_scales():
    assert adaptive_bar_height(2) == 480
    assert adaptive_bar_height(20) > adaptive_bar_height(5)
    assert adaptive_bar_height(100) == 1100
