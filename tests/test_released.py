"""Adversarial tests for post-competition label and support-example isolation."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.released import (
    download_release,
    export_release,
    load_plan,
    load_research,
    metric_attachment_audit,
    partition_release,
    prepare_release,
    read_research_targets,
    released_evidence,
)
from jigsaw_rules.runtime import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def boundary():
    plan = load_plan(ROOT)
    rows, metadata = [], []
    for i, (name, rule) in enumerate(plan["rule_text"].items()):
        for usage in ("Public", "Private"):
            row_id = i * 10 + int(usage == "Private")
            rows.append(
                {
                    "row_id": row_id,
                    "rule": rule,
                    "body": f"comment {row_id}",
                    "subreddit": "test",
                    **{col: f"support {col}" for col in EXAMPLES},
                }
            )
            metadata.append({"row_id": row_id, "rule_id": name, "Usage": usage})
    historical = pd.DataFrame([{**rows[0], "body": "historical comment"}])
    return historical, pd.DataFrame(rows), pd.DataFrame(metadata), plan


def test_boundary_retains_two_whole_policies_and_every_private_row(boundary):
    assignments, report = partition_release(*boundary)
    assert len(assignments) == report["rows"] == 12
    assert assignments.loc[assignments.Usage.eq("Private"), "role"].eq("confirmation").all()
    assert (
        assignments.loc[assignments.rule_id.isin(["financial", "spoilers"]), "role"]
        .eq("confirmation")
        .all()
    )
    assert report["role_counts"] == {"confirmation": 8, "research": 4}
    assert not report["assignment_uses_targets"]


@pytest.mark.parametrize("column", ["body", *EXAMPLES])
def test_any_research_context_copy_of_reserved_body_is_purged(boundary, column):
    historical, inputs, metadata, plan = boundary
    inputs.loc[inputs.row_id.eq(0), column] = "  COMMENT\n1  "
    assignments, report = partition_release(historical, inputs, metadata, plan)
    assert assignments.set_index("row_id").loc[0, "role"] == "boundary_overlap"
    assert report["research_context_confirmation_body_overlap"] == 0


def test_historical_support_exposure_cannot_masquerade_as_independent_comment(boundary):
    historical, inputs, metadata, plan = boundary
    historical.loc[0, "positive_example_1"] = "ＣＯＭＭＥＮＴ 1"
    assignments, _ = partition_release(historical, inputs, metadata, plan)
    assert assignments.set_index("row_id").loc[1, "role"] == "historical_overlap"


def test_assignment_is_order_invariant_and_metadata_join_is_by_id(boundary):
    historical, inputs, metadata, plan = boundary
    first, counts = partition_release(*boundary)
    second, other = partition_release(
        historical, inputs.iloc[::-1], metadata.sample(frac=1, random_state=5), plan
    )
    pd.testing.assert_frame_equal(first, second)
    assert counts == other


@pytest.mark.parametrize("problem", ["target", "duplicate", "missing", "usage", "rule"])
def test_partition_rejects_ambiguous_or_target_bearing_metadata(boundary, problem):
    historical, inputs, metadata, plan = boundary
    if problem == "target":
        metadata["rule_violation"] = 0
    elif problem == "duplicate":
        metadata.loc[1, "row_id"] = metadata.loc[0, "row_id"]
    elif problem == "missing":
        metadata = metadata.iloc[1:]
    elif problem == "usage":
        metadata.loc[0, "Usage"] = "Unknown"
    else:
        inputs.loc[0, "rule"] = "changed rule"
    with pytest.raises(ValueError):
        partition_release(historical, inputs, metadata, plan)


def write_solution(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["row_id", "Usage", "rule_violation", "rule"])
        writer.writeheader()
        writer.writerows(rows)


def test_reserved_targets_are_not_interpreted_or_returned(tmp_path):
    path = tmp_path / "solution.csv"
    rows = [
        {"row_id": 0, "Usage": "Public", "rule_violation": "1", "rule": "medical"},
        {"row_id": 1, "Usage": "Private", "rule_violation": "NEVER_PARSE_THIS", "rule": "medical"},
        {"row_id": 2, "Usage": "Public", "rule_violation": "NEVER_PARSE_THIS", "rule": "financial"},
    ]
    write_solution(path, rows)
    result = read_research_targets(path, {0}, {"medical"})
    assert result.to_dict("records") == [{"row_id": 0, "rule_violation": 1}]
    rows[1]["rule_violation"] = "changed protected value"
    write_solution(path, rows)
    pd.testing.assert_frame_equal(result, read_research_targets(path, {0}, {"medical"}))
    for ids in ({1}, {2}):
        with pytest.raises(ValueError, match="public and binary"):
            read_research_targets(path, ids, {"medical"})


def test_missing_or_invalid_research_targets_fail_closed(tmp_path):
    path = tmp_path / "solution.csv"
    write_solution(
        path, [{"row_id": 0, "Usage": "Public", "rule_violation": "0.2", "rule": "medical"}]
    )
    with pytest.raises(ValueError, match="binary"):
        read_research_targets(path, {0}, {"medical"})
    with pytest.raises(ValueError, match="Missing"):
        read_research_targets(path, {99}, {"medical"})


def test_metric_audit_reports_mismatches_and_missing_rows():
    scores = pd.DataFrame(
        {"PrivateScore": [0.8, 0.8, 0.5], "a": [0.7, 0.1, None], "b": [0.9, 0.2, 0.5]}
    )
    result = metric_attachment_audit(scores, ["a", "b"], 1e-6)
    assert result["matching_rows"] == 1
    assert result["discrepant_complete_rows"] == 1
    assert result["incomplete_rows"] == 1


def test_download_reuses_verified_members_and_rejects_modified_sources(tmp_path, monkeypatch):
    import hashlib

    folder = tmp_path / "data/released/v1"
    folder.mkdir(parents=True)
    payload = b"row_id,body\n0,example\n"
    (folder / "test.csv").write_bytes(payload)
    plan = {
        "source": {
            "files": {
                "test.csv": {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            }
        }
    }

    def no_network(*args, **kwargs):
        raise AssertionError("Verified files must not be downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    assert download_release(tmp_path, plan) == folder
    (folder / "test.csv").write_bytes(payload + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        download_release(tmp_path, plan)


def test_public_evidence_rejects_unbound_file_allowlist(tmp_path):
    folder = tmp_path / "reports/released"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(
        json.dumps({"schema": 1, "files": {"../../private.csv": "abc"}})
    )
    with pytest.raises(ValueError, match="contract"):
        released_evidence(tmp_path)


@pytest.fixture
def prepared_project(tmp_path, boundary):
    historical, inputs, metadata, plan = boundary
    folder = tmp_path / "data/released/v1"
    original = tmp_path / "data/raw"
    folder.mkdir(parents=True)
    original.mkdir(parents=True)
    historical.assign(rule_violation=0).to_csv(original / "train.csv", index=False)
    historical.assign(row_id=999).to_csv(original / "test.csv", index=False)
    (folder / "train.csv").write_bytes((original / "train.csv").read_bytes())
    (folder / "public_test.csv").write_bytes((original / "test.csv").read_bytes())
    inputs.to_csv(folder / "test.csv", index=False)
    rows = [
        {
            "row_id": row.row_id,
            "Usage": row.Usage,
            "rule_violation": "0"
            if row.Usage == "Public" and row.rule_id in plan["research_rule_ids"]
            else "DO_NOT_PARSE",
            "rule": row.rule_id,
        }
        for row in metadata.itertuples()
    ]
    write_solution(folder / "solution.csv", rows)
    plan["source"]["files"] = {
        p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in folder.glob("*.csv")
    }
    attachment = tmp_path / "reports/private/official-rule-auc.csv"
    attachment.parent.mkdir(parents=True)
    pd.DataFrame([{"PrivateScore": 0.5, **{name: 0.5 for name in plan["rule_text"]}}]).to_csv(
        attachment, index=False
    )
    plan["official_metric_evidence"]["sha256"] = digest(attachment)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/released.json").write_text(json.dumps(plan))
    directory = prepare_release(tmp_path)
    export_release(tmp_path, directory)
    return tmp_path, directory


def test_complete_pipeline_exposes_only_research_and_reuses_stage(prepared_project, monkeypatch):
    root, directory = prepared_project
    research = load_research(root)
    assert len(research) == 4
    before = (directory / "complete.json").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed boundary should not reinterpret targets")

    monkeypatch.setattr("jigsaw_rules.released.read_research_targets", forbidden)
    assert prepare_release(root) == directory
    assert (directory / "complete.json").read_bytes() == before
    (directory / "research.csv").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        load_research(root)


def test_changed_plan_cannot_relabel_an_old_boundary(prepared_project):
    root, directory = prepared_project
    path = root / "configs/released.json"
    plan = json.loads(path.read_text())
    plan["target_access"] = "modified protocol"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="plan"):
        export_release(root, directory)
    with pytest.raises(ValueError, match="contract"):
        load_research(root)
