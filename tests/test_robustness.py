import pandas as pd

from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.robustness import near_copy_edges, purge_near


def test_near_copy_requires_both_similarity_conditions():
    sentence = "This is a sufficiently lengthy example of a repeated discussion comment."
    rows, columns = near_copy_edges([sentence, "ok", "unrelated topic"], [sentence + "!", "ok"])
    # Punctuation changes also change one of twelve whitespace tokens: below 0.90 Jaccard.
    assert rows.size == columns.size == 0
    rows, columns = near_copy_edges([sentence], [sentence.upper() + "  "])
    assert rows.tolist() == columns.tolist() == [0]


def test_purge_finds_copies_in_any_support_column_and_ignores_targets():
    sentence = "This is a long comment repeated as an example across a validation boundary."
    training = pd.DataFrame(
        {
            "body": ["safe comment", "other comment"],
            **{column: ["short", "short"] for column in EXAMPLES},
        }
    )
    training.loc[1, EXAMPLES[-1]] = sentence
    validation = pd.DataFrame({"body": [sentence], "rule_violation": [0]})
    retained, affected = purge_near(training, validation)
    validation.rule_violation = 1
    repeated, _ = purge_near(training, validation)
    assert retained.index.tolist() == repeated.index.tolist() == [0]
    assert affected.tolist() == [0]
