from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from jigsaw_rules.data import EXAMPLES
from jigsaw_rules.runtime import atomic_json, digest
from scripts import feature_value_audit as fa
from scripts import run_feature_value_audit as run
from scripts.run_behavioral_features import protocol, stage_write


def fixture_frame():
    rows = []
    for policy, name in enumerate(("No Advertising", "No legal advice")):
        for i in range(48):
            body = f"authored item {policy} {i}"
            if i % 3 == 0:
                body += ' he said "you should proceed"'
            if i % 5 == 0:
                body += " should I not try? https://example.test"
            row = {
                "row_id": policy * 100 + i,
                "body": body,
                "rule": name,
                "subreddit": "authored",
                "rule_violation": i % 2,
            }
            row.update({column: f"support {name} {column}" for column in EXAMPLES})
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    source = Path(__file__).parents[1]
    for directory in ("scripts", "src", "tests", "configs", "docs", "notebooks"):
        shutil.copytree(source / directory, root / directory)
    config = json.loads((root / run.CONFIG).read_text())
    frame = fixture_frame()
    raw = root / "data/raw/train.csv"
    raw.parent.mkdir(parents=True)
    frame.to_csv(raw, index=False)
    config.update(
        train_sha256=digest(raw),
        query_counts=[48, 48],
        bootstrap_replicates=40,
        minimum_slice_class=3,
    )
    plan, cohorts = protocol(frame, [48, 48])
    rng = np.random.default_rng(18)
    predictions = {}
    for i, fold in enumerate(plan["folds"]):
        ids = np.array([row["row_id"] for row in fold["queries"]])
        y = frame.set_index("row_id").loc[ids].rule_violation.to_numpy()
        margin = y * 0.8 + rng.normal(size=len(y))
        path = root / fa.PRIVATE / "reference" / f"fold_{i}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            query_row_ids=ids,
            query_adapted_scores=np.column_stack([expit(margin), margin, np.ones(len(y))]),
        )
        config["reference_files"][i].update(bytes=path.stat().st_size, sha256=digest(path))
        config["reference_auc"][i] = float(roc_auc_score(y, margin))
        for name in fa.VARIANTS:
            predictions[(i, name)] = (ids, y, expit(y * 0.5 + rng.normal(size=len(y))))
    titles = {1: "add_behavior vs control", 2: "add_act_roles vs behavior"}
    comparator = {
        3: ("condition_lexical", "copy_lexical"),
        4: ("rule_both", "permuted_both"),
        5: ("scope_both", "copy_both"),
    }
    for item in config["rounds"]:
        number, metrics = item["round"], []
        ident = {
            "source": {
                "scripts/feature_value_audit.py": digest(root / "scripts/feature_value_audit.py")
            }
        }
        for name in fa.VARIANTS:
            if config["model_round"][name] != number:
                continue
            for i in range(2):
                ids, y, p = predictions[(i, name)]
                data = io.BytesIO()
                np.savez(data, row_ids=ids, probability=p)
                folder = root / item["private"] / item["run_id"] / f"fold_{i}" / name
                key = {"run_id": item["run_id"], "fold": i, "variant": name}
                stage_write(folder, {"predictions.npz": data.getvalue()}, key)
                metrics.append({"fold": i, "variant": name, "auc": float(roc_auc_score(y, p))})
        contrast = {
            "comparison": titles.get(number, "synthetic matched comparison"),
            "delta_auc": 0.01,
            "simultaneous_low": -0.01,
            "simultaneous_high": 0.03,
        }
        if number in comparator:
            contrast.update(candidate=comparator[number][0], reference=comparator[number][1])
        report = {
            "run_id": item["run_id"],
            "identity": ident,
            "metrics": metrics,
            "cohorts": cohorts,
            "comparisons": [contrast],
            "decision": "NOT_PROMOTED_AUTHORED_TEST",
        }
        path = root / item["public"] / "results.json"
        atomic_json(path, report)
        item["sha256"] = digest(path)
        atomic_json(
            root / item["private"] / item["run_id"] / "finished.json",
            {
                "identity": ident,
                "public_hashes": {"results.json": digest(path)},
            },
        )
    atomic_json(root / run.CONFIG, config)
    return root


@pytest.mark.parametrize("bad", [[0, 0], [1, 1], [0, 2], [0, 0.5], [0, np.nan]])
def test_nonbinary_or_one_class_labels_stop(bad):
    with pytest.raises(ValueError):
        fa.binary_labels(bad)


@pytest.mark.parametrize("bad", [[0, np.nan], [0, np.inf], [0, -0.1], [0, 1.1], [[0, 1]]])
def test_invalid_scores_stop(bad):
    with pytest.raises(ValueError):
        fa.probabilities(bad, 2)


@pytest.mark.parametrize("bad", [[1.0, 2.0], [1, 1], [2, 1], [1, 3]])
def test_bad_id_alignment_stops(bad):
    with pytest.raises(ValueError):
        fa.ordered_ids(np.array(bad), np.array([1, 2]))


def test_tie_credit_and_auc_decomposition():
    y = np.array([0, 0, 1, 1])
    r = np.array([0.5, 0.5, 0.5, 0.5])
    a = np.array([0.1, 0.7, 0.6, 0.9])
    out = fa.pair_decomposition(y, r, a)
    assert out["reference_ties"] == 4
    assert out["net_auc_delta"] == pytest.approx(roc_auc_score(y, a) - 0.5)
    assert out["gained_auc_credit"] - out["lost_auc_credit"] == pytest.approx(out["net_auc_delta"])


def test_identical_predictions_have_no_gain_or_damage():
    out = fa.pair_decomposition([0, 1, 0, 1], [0.2, 0.8, 0.7, 0.1], [0.2, 0.8, 0.7, 0.1])
    assert out["gained_auc_credit"] == out["lost_auc_credit"] == 0


def test_reference_ranks_margins_not_saturated_probabilities(tmp_path):
    path = tmp_path / "scores.npz"
    ids = np.arange(4)
    margins = np.array([20.0, 30.0, 40.0, 50.0])
    # GPU float32 probability saturation must not erase relative order.
    np.savez(
        path,
        query_row_ids=ids,
        query_adapted_scores=np.column_stack([np.ones(4), margins, np.ones(4)]),
    )
    ranks, p = fa.read_reference(path, ids)
    np.testing.assert_array_equal(ranks, (rankdata(margins) - 0.5) / 4)
    np.testing.assert_allclose(p, expit(margins))


def test_slice_builder_refuses_targets():
    frame = fixture_frame().iloc[:4]
    with pytest.raises(ValueError, match="target-free"):
        fa.slice_flags(frame)
    flags = fa.slice_flags(frame[["row_id", "body", "rule"]])
    assert set(flags) == set(fa.SLICES)


def test_constant_correlation_is_none():
    assert fa.rank_correlation([1, 1], [0, 1]) is None


def test_ledger_reads_all_actual_historical_report_schemas():
    source = Path(__file__).parents[1]
    config = json.loads((source / run.CONFIG).read_text())
    reports = {
        r["round"]: (json.loads((source / r["public"] / "results.json").read_text()), None)
        for r in config["rounds"]
    }
    rows = run.ledger(reports)
    assert len(rows) == 5
    assert rows[-1]["delta_auc"] == pytest.approx(-0.0003785297013493283)


def test_end_to_end_no_fits_network_or_previous_mutation(project, monkeypatch):
    import boto3
    from sklearn.linear_model import LogisticRegression

    def forbidden(*args, **kwargs):
        raise AssertionError("network/model fit unexpectedly invoked")

    monkeypatch.setattr(boto3, "client", forbidden)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    prior = [p for p in (project / "reports").glob("*/results.json")]
    hashes = {p: digest(p) for p in prior}
    result = run.run_audit(project, home=project.parent)
    assert result["model_fits"] == result["new_neural_inference"] == result["blends_tested"] == 0
    assert result["promotion"] == "NONE_DIAGNOSTIC_ONLY"
    assert len(result["metrics"]) == 16
    assert len(result["contrasts"]) == 7
    assert len(result["research_ledger"]) == 5
    assert len(run.figures(result)) == 8
    assert all(digest(p) == h for p, h in hashes.items())


def test_replay_does_not_recompute_statistics(project, monkeypatch):
    first = run.run_audit(project, home=project.parent)
    monkeypatch.setattr(fa, "summarize", lambda *a, **k: pytest.fail("statistics recomputed"))
    assert run.run_audit(project, home=project.parent) == first


def test_corrupt_completed_output_rejected(project):
    run.run_audit(project, home=project.parent)
    (project / fa.PUBLIC / "audit.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        run.run_audit(project, home=project.parent)


def test_corrupt_private_prediction_rejected(project):
    config = json.loads((project / run.CONFIG).read_text())
    item = config["rounds"][0]
    p = project / item["private"] / item["run_id"] / "fold_0/lexical_control/predictions.npz"
    p.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        run.run_audit(project, home=project.parent)


def test_reference_metric_reconciliation_stops(project):
    config = json.loads((project / run.CONFIG).read_text())
    config["reference_auc"] = [0.999, 0.999]
    atomic_json(project / run.CONFIG, config)
    with pytest.raises(ValueError, match="published development AUC"):
        run.run_audit(project, home=project.parent)


def test_small_slices_not_scored_as_zero(project):
    result = run.run_audit(project, home=project.parent)
    empty = [r for r in result["slices"] if r["rows"] == 0]
    assert empty and all(r["auc"] is None for r in empty)


def test_export_excludes_private_files_and_raw_text(project, monkeypatch, tmp_path):
    run.run_audit(project, home=project.parent)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    out = run.export(project)
    with zipfile.ZipFile(out) as z:
        assert not any(n.endswith((".npz", ".csv")) or n.startswith("data/") for n in z.namelist())
        for name, value in json.loads(z.read("SHA256SUMS.json")).items():
            assert hashlib.sha256(z.read(name)).hexdigest() == value
        aggregate = z.read(fa.PUBLIC + "/audit.json").decode()
        assert "authored item" not in aggregate


class Log:
    def emit(self, *args, **kwargs):
        pass


class FakeSTS:
    def __init__(self, account):
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account}


class FakeS3:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        data = self.payloads[kwargs["Key"]]
        return {"ContentLength": len(data), "Body": io.BytesIO(data)}


def recovery_setup(tmp_path):
    root, home = tmp_path / "root", tmp_path / "home"
    root.mkdir()
    home.mkdir()
    payloads = {"first": b"alpha", "second": b"beta"}
    config = {
        "account": "123",
        "region": "us-west-2",
        "bucket": "private-test",
        "reference_files": [],
    }
    for i, (key, data) in enumerate(payloads.items()):
        config["reference_files"].append(
            {
                "name": f"fold_{i}.npz",
                "key": key,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return root, home, config, payloads


def test_explicit_s3_recovery_read_once_per_missing_file(tmp_path):
    root, home, config, payloads = recovery_setup(tmp_path)
    s3 = FakeS3(payloads)
    paths, receipts = fa.recover_reference(
        root, home, config, allow_s3=True, log=Log(), clients=(FakeSTS("123"), s3)
    )
    assert len(s3.calls) == 2 and all(p.is_file() for p in paths)
    fa.recover_reference(root, home, config, allow_s3=False, log=Log())
    assert len(s3.calls) == 2
    assert all(r["origin"] == "verified_private_s3_read" for r in receipts)


def test_wrong_account_prevents_s3_reads(tmp_path):
    root, home, config, payloads = recovery_setup(tmp_path)
    s3 = FakeS3(payloads)
    with pytest.raises(ValueError, match="AWS account"):
        fa.recover_reference(
            root, home, config, allow_s3=True, log=Log(), clients=(FakeSTS("wrong"), s3)
        )
    assert not s3.calls


def test_s3_failure_preserves_first_completed_file(tmp_path):
    root, home, config, payloads = recovery_setup(tmp_path)
    config["reference_files"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        fa.recover_reference(
            root, home, config, allow_s3=True, log=Log(), clients=(FakeSTS("123"), FakeS3(payloads))
        )
    assert (root / fa.PRIVATE / "reference/fold_0.npz").read_bytes() == b"alpha"
    assert not (root / fa.PRIVATE / "reference/fold_1.npz").exists()


def test_corrupt_local_cache_never_redownloaded(tmp_path):
    root, home, config, payloads = recovery_setup(tmp_path)
    target = root / fa.PRIVATE / "reference/fold_0.npz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bad")
    s3 = FakeS3(payloads)
    with pytest.raises(ValueError, match="corrupt"):
        fa.recover_reference(
            root, home, config, allow_s3=True, log=Log(), clients=(FakeSTS("123"), s3)
        )
    assert not s3.calls and target.read_bytes() == b"bad"


def test_no_reference_without_recovery_permission(tmp_path):
    root, home, config, _ = recovery_setup(tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        fa.recover_reference(root, home, config, allow_s3=False, log=Log())


def test_preserved_local_copies_reused_without_network(tmp_path):
    root, home, config, payloads = recovery_setup(tmp_path)
    for i, data in enumerate(payloads.values()):
        folder = home / "jigsaw-preserved" / f"old/runs/fold_{i}"
        folder.mkdir(parents=True)
        (folder / "representations.npz").write_bytes(data)
    paths, receipts = fa.recover_reference(root, home, config, allow_s3=False, log=Log())
    assert len(paths) == 2
    assert all(r["origin"] == "verified_existing_private_copy" for r in receipts)
