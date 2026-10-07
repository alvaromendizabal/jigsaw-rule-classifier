"""Reject stale, partial and misleading employer-facing evidence."""

import json
import shutil
from pathlib import Path

import nbformat
import pytest

from scripts.review_portfolio import INPUTS, MIME, NOTEBOOK, ROOT, evidence, verify


def test_saved_complete_review_matches_aggregate_receipts():
    result = verify(nbformat.read(ROOT / NOTEBOOK, as_version=4))
    assert result["private_auc"] == 0.91425
    assert result["plotly"] == result["svg"] == 2


def test_saved_frontier_has_real_values_and_matching_fallback():
    from scripts.review_portfolio import FRONTIER_NOTEBOOK

    result = verify(nbformat.read(ROOT / FRONTIER_NOTEBOOK, as_version=4))
    assert result["plotly"] == result["svg"] == 1


@pytest.mark.parametrize(
    "mutation", ["source", "summary", "plot", "svg", "geometry", "error", "partial"]
)
def test_saved_review_rejects_corruption(mutation):
    notebook = nbformat.read(ROOT / NOTEBOOK, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    plot = next(o for c in code for o in c.outputs if MIME in o.get("data", {}))
    if mutation == "source":
        code[0].source += "\nprint('unreviewed')"
    elif mutation == "summary":
        code[-1].outputs = []
    elif mutation == "plot":
        plot.data[MIME]["data"][0]["x"][0] = 0.99
    elif mutation == "svg":
        del plot.data["image/svg+xml"]
    elif mutation == "geometry":
        import re

        plot.data["image/svg+xml"] = re.sub(
            r'width="[0-9.]+" height="27"', 'width="1.00" height="27"', plot.data["image/svg+xml"]
        )
    elif mutation == "error":
        code[-1].outputs.append(nbformat.v4.new_output("stream", name="stderr", text="failed"))
    else:
        code[-1].execution_count = None
    with pytest.raises(ValueError):
        verify(notebook)


@pytest.fixture
def evidence_root(tmp_path):
    for relative in INPUTS:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return tmp_path


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, -0.1, 1.1])
def test_official_evidence_rejects_invalid_auc(evidence_root, value):
    path = evidence_root / INPUTS[0]
    data = json.loads(path.read_text())
    data["submission"]["public_score"] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        evidence(evidence_root)


def test_receipt_change_invalidates_saved_output(evidence_root):
    path = evidence_root / INPUTS[2]
    data = json.loads(path.read_text())
    data["generated_utc"] = "2026-10-07T00:00:00Z"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inputs changed"):
        verify(nbformat.read(ROOT / NOTEBOOK, as_version=4), evidence_root)


def test_reciprocal_control_cannot_be_substituted(evidence_root):
    path = evidence_root / INPUTS[2]
    data = json.loads(path.read_text())
    reciprocal = next(s for s in data["studies"] if s["id"] == "E31")
    reciprocal["matched_native_control_auc"] = 0.1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="arithmetic"):
        evidence(evidence_root)


def test_untouched_holdout_claim_is_rejected(evidence_root):
    path = evidence_root / INPUTS[2]
    data = json.loads(path.read_text())
    data["evaluation"]["untouched_holdout"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="scope"):
        evidence(evidence_root)


def test_review_runs_without_private_inputs():
    summary = evidence()
    assert summary["development_cohort"] == {"rows": 881, "policies": 2}
    assert summary["next_study"] == {"id": "E44", "status": "planned_in_aws"}
    assert all(Path(p).parts[0] in {"reports", "scripts"} for p in INPUTS)
