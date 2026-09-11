from __future__ import annotations

import hashlib
import io
import json
import tarfile

from scripts.asymmetric_support_processing import extract_plan, fixed_review_cases


def test_fixed_review_cases_are_deterministic_and_target_free():
    rule = "No advertising"
    fold = {
        "rule": rule,
        "training": [
            {"body": "positive", "rule": rule, "rule_violation": 1, "repeat": 2},
            {"body": "negative", "rule": rule, "rule_violation": 0, "repeat": 2},
        ],
        "queries": [
            {"row_id": i, "body": f"query {i}", "rule": rule}
            for i in range(10, 16)
        ],
    }
    rows = []
    for i, query in enumerate(fold["queries"]):
        method = {
            "positive": {"training_index": 0, "support_id": "p", "cosine": 0.8},
            "negative": {"training_index": 1, "support_id": "n", "cosine": -0.2},
        }
        rows.append(
            {
                "query_index": i,
                "row_id": query["row_id"],
                "lexical": method,
                "semantic": method,
                "asymmetric": method,
            }
        )
    first = fixed_review_cases(fold, rows, sample_size=3, seed=2025, fold_index=0)
    second = fixed_review_cases(fold, rows, sample_size=3, seed=2025, fold_index=0)
    assert first == second
    assert len(first) == 3
    assert all(case["query_target_included"] is False for case in first)
    assert all(set(case["methods"]) == {"lexical", "semantic", "asymmetric"} for case in first)


def test_extract_plan_reads_only_hash_pinned_regular_member(tmp_path):
    payload = json.dumps({"schema": 1, "protocol": "new_rule_supplied_support_adaptation", "folds": []}).encode()
    source = tmp_path / "source.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        info = tarfile.TarInfo("adaptation_plan.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    destination = tmp_path / "plan.json"
    try:
        extract_plan(
            source,
            {
                "member": "adaptation_plan.json",
                "max_bytes": 1024,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            destination,
        )
    except ValueError as exc:
        # The member hash/extraction is valid; the intentionally empty synthetic plan then fails plan validation.
        assert "At least one fold" in str(exc)
    else:
        raise AssertionError("empty synthetic plan should fail reviewed plan validation")
    assert destination.read_bytes() == payload
