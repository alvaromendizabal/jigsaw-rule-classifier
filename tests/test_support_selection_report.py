"""Publication checks use saved public aggregates only, never query examples or targets."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import build_support_selection_report as report

ROOT = Path(__file__).resolve().parents[1]


def fixture_root(tmp_path):
    shutil.copytree(ROOT / "reports/support_selection", tmp_path / "reports/support_selection")
    return tmp_path


def test_saved_evidence_and_exact_counts():
    saved = report.evidence(ROOT)
    rows, reviews = report.tables(saved)
    assert sum(r["Changed pairs"] for r in rows) == 878
    assert rows[1]["Semantic max permitted-example reuse"] == 121
    assert rows[1]["Lexical max permitted-example reuse"] == 21
    assert sum(r["lexical"] for r in reviews) == 24
    assert saved["audit"]["status"] == "awaiting_blinded_relevance_review"
    assert saved["review_decision"]["new_auc"] is None
    csls = report.csls_table(saved)
    assert [row["Changed pairs"] for row in csls] == [114, 268]
    assert csls[1]["CSLS negative max reuse (%)"] > csls[1]["Raw negative max reuse (%)"]
    assert all(row["Gate passed"] is False for row in csls)


@pytest.mark.parametrize("name", sorted(report.EXPECTED))
def test_each_payload_is_hash_checked(tmp_path, name):
    root = fixture_root(tmp_path)
    p = root / "reports/support_selection" / name
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        report.evidence(root)


def test_unknown_manifest_member_rejected(tmp_path):
    root = fixture_root(tmp_path)
    p = root / "reports/support_selection/metadata.json"
    doc = json.loads(p.read_text())
    doc["files"]["../../private.json"] = "0" * 64
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest"):
        report.evidence(root)


def test_false_metric_rejected_even_with_rehashed_payload(tmp_path):
    root = fixture_root(tmp_path)
    p = root / "reports/support_selection/audit.json"
    doc = json.loads(p.read_text())
    doc["official_metric"] = 0.99
    p.write_text(json.dumps(doc))
    m = p.parent / "metadata.json"
    manifest = json.loads(m.read_text())
    manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="AUC"):
        report.evidence(root)


def test_cells_are_read_only_and_explicit_about_review_limits():
    cells = report.notebook_cells()
    assert len(cells) == 6
    narrative = "\n".join(text for kind, text in cells if kind == "md")
    assert "one AI reviewer" in narrative
    assert "CSLS" in narrative
    assert "18.70% to 19.63%" in narrative
    assert "No new AUC" in narrative
    code = "\n".join(text for kind, text in cells if kind == "code")
    assert not any(s in code for s in ("torch", "fit(", "audit_support_selection", "np.load"))
    compile(code, "notebook-section", "exec")


def test_builder_inserts_section_once_in_both_canonical_notebooks():
    from scripts.build_notebooks import notebooks

    generated = notebooks()
    for name in ("02_baseline_and_review", "03_saved_results"):
        nb = generated[f"notebooks/{name}.ipynb"]
        assert sum("Latest diagnostic: cached-vector" in c.source for c in nb.cells) == 1
        assert sum("hubness-corrected semantic selection" in c.source for c in nb.cells) == 1
        assert len({c.id for c in nb.cells}) == len(nb.cells)


def test_execution_contract_includes_new_sources_and_evidence():
    source = (ROOT / "scripts/execute_notebooks.py").read_text()
    assert '"build_support_selection_report.py"' in source
    assert '"support_selection.json"' in source
    assert '"support_selection"' in source


def test_tables_do_not_mutate_original_evidence():
    saved = report.evidence(ROOT)
    before = copy.deepcopy(saved)
    report.tables(saved)
    assert saved == before


def test_historical_delivery_still_checks_original_source_bytes(tmp_path):
    from scripts.build_delivery_report import delivery_evidence

    # A new, separate research module cannot invalidate a frozen release display.
    assert delivery_evidence(ROOT)["models_refitted"] == 0
    for name in ("src/jigsaw_rules", "reports/delivery", "configs"):
        shutil.copytree(ROOT / name, tmp_path / name)
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/verify_offline.py", tmp_path / "scripts/verify_offline.py")
    assert delivery_evidence(tmp_path)["status"] == "passed"
    source = tmp_path / "src/jigsaw_rules/data.py"
    source.write_bytes(source.read_bytes() + b"\n# altered historical source\n")
    with pytest.raises(ValueError, match="implementation"):
        delivery_evidence(tmp_path)
