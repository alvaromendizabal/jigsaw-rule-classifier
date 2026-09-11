"""Authored fixtures only. These tests do not estimate competition performance."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from jigsaw_rules import support_selection as s
from jigsaw_rules.data import normalize
from scripts import audit_support_selection as audit
from scripts import recover_support_selection as recovery
from scripts import support_selection_runtime as rt

ROOT = Path(__file__).resolve().parents[1]


def authored_plan():
    folds = []
    for fi, rule in enumerate(("No advertisements", "No legal advice")):
        bodies = (
            "Zeta buy discount offer",
            "Alpha discussion of courts",
            "Beta you should file a claim",
            "Gamma can I ask a question?",
        )
        training = [
            {
                "body": f"{b} source{fi}",
                "rule": rule,
                "rule_violation": int(i % 2 == 0),
                "repeat": 2,
            }
            for i, b in enumerate(bodies)
        ]
        training.insert(
            1,
            {
                "body": f"Other policy support {fi}",
                "rule": "Other rule",
                "rule_violation": 1,
                "repeat": 1,
            },
        )
        queries = [
            {
                "row_id": 100 * fi + i,
                "body": f"Private can I buy discount question{fi}-{i}?",
                "rule": rule,
            }
            for i in range(5)
        ]
        folds.append({"rule": rule, "training": training, "queries": queries, "audit": {}})
    return {"schema": 1, "protocol": s.PROTOCOL, "folds": folds}


def vector_values(fold, dimensions=8):
    rng = np.random.default_rng(445)
    return (
        rng.normal(size=(len(fold["training"]), dimensions)).astype("float32"),
        rng.normal(size=(len(fold["queries"]), dimensions)).astype("float32"),
    )


def write_vectors(path, fold, train=None, query=None, **extras):
    default_train, default_query = vector_values(fold)
    np.savez_compressed(
        path,
        train_adapted_vectors=default_train if train is None else train,
        query_adapted_vectors=default_query if query is None else query,
        query_row_ids=[r["row_id"] for r in fold["queries"]],
        **extras,
    )


def make_archive(path, payload, member="adaptation_plan.json", duplicate=False, link=False):
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        if link:
            info.type = tarfile.SYMTYPE
            info.linkname = "../forbidden"
            info.size = 0
        archive.addfile(info, None if link else io.BytesIO(payload))
        if duplicate:
            archive.addfile(info, io.BytesIO(payload))
        other = tarfile.TarInfo("../../outside.py")
        other.size = 9
        archive.addfile(other, io.BytesIO(b"forbidden"))


@pytest.fixture
def fixture(tmp_path):
    plan = authored_plan()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = rt.json_bytes(plan)
    (inputs / "plan.json").write_bytes(payload)
    make_archive(inputs / "source.tar.gz", payload)
    config = json.loads((ROOT / "configs/support_selection.json").read_text())
    config["evidence_scope"] = "authored synthetic software tests; NOT competition evidence"
    config["dimensions"] = 8
    config["selection"]["batch_size"] = 2
    config["audit"]["sample_per_policy"] = 3
    config["plan"]["sha256"] = rt.sha_bytes(payload)
    config["expected_counts"] = [{"training_rows": 5, "query_rows": 5, "same_rule_supports": 4}] * 2
    for fi, fold in enumerate(plan["folds"]):
        write_vectors(inputs / f"fold_{fi}.npz", fold, query_adapted_scores=np.full((5, 2), np.nan))
    for pin in [config["source"], *config["representations"]]:
        path = inputs / pin["name"]
        pin.update(bytes=path.stat().st_size, sha256=rt.digest(path))
    config_path = tmp_path / "config.json"
    config_path.write_bytes(rt.json_bytes(config))
    return {
        "plan": plan,
        "inputs": inputs,
        "config": config,
        "config_path": config_path,
        "plan_path": inputs / "plan.json",
        "output": tmp_path / "audit",
    }


def run(f):
    return audit.run_audit(f["config_path"], f["plan_path"], f["inputs"], f["output"], 30, 1)


def test_plan_counts():
    checks = s.validate_plan(authored_plan())
    assert len(checks) == 2
    assert all(c["same_rule_supports"] == 4 for c in checks)
    assert checks[0]["support_indices"] == [0, 2, 3, 4]


@pytest.mark.parametrize("field", ["rule_violation", "target", "prediction", "score"])
def test_reject_query_extras(field):
    plan = authored_plan()
    plan["folds"][0]["queries"][0][field] = 0
    with pytest.raises(ValueError, match="Query schema"):
        s.validate_plan(plan)


@pytest.mark.parametrize("other_rule", [True, False])
def test_purge_all_sources_within_fold(other_rule):
    plan = authored_plan()
    fold = plan["folds"][0]
    row = fold["training"][1 if other_rule else 0]
    row["body"] = "  " + fold["queries"][0]["body"].upper() + "\n"
    with pytest.raises(ValueError, match="Query body"):
        s.validate_plan(plan)


def test_does_not_purge_legitimate_crossfold_training():
    plan = authored_plan()
    # Another fold's query may legitimately be an original training row here.
    fold = plan["folds"][0]
    fold["training"].append(
        {
            **{k: v for k, v in plan["folds"][1]["queries"][0].items() if k != "row_id"},
            "rule_violation": 0,
            "repeat": 1,
        }
    )
    assert s.validate_plan(plan)[0]["training_rows"] == 6


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_pair(conflicting):
    plan = authored_plan()
    row = copy.deepcopy(plan["folds"][0]["training"][0])
    row["body"] = row["body"].upper()
    if conflicting:
        row["rule_violation"] = 1 - row["rule_violation"]
    plan["folds"][0]["training"].append(row)
    with pytest.raises(ValueError, match="Duplicate or conflicting"):
        s.validate_plan(plan)


def test_provenance_repeat_guard():
    plan = authored_plan()
    plan["folds"][0]["training"][0]["repeat"] = 1
    with pytest.raises(ValueError, match="provenance"):
        s.validate_plan(plan)


def test_ids_unique_across_folds():
    plan = authored_plan()
    plan["folds"][1]["queries"][0]["row_id"] = 0
    with pytest.raises(ValueError, match="unique"):
        s.validate_plan(plan)


def test_read_only_three_arrays(fixture, monkeypatch):
    real = np.lib.npyio.NpzFile.__getitem__
    loaded = []

    def guarded(self, name):
        assert name in s.READ_ARRAYS
        loaded.append(name)
        return real(self, name)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guarded)
    train, query = s.load_vectors(fixture["inputs"] / "fold_0.npz", fixture["plan"]["folds"][0], 8)
    assert loaded == list(s.READ_ARRAYS)
    assert train.shape == query.shape == (5, 8)


@pytest.mark.parametrize("problem", ["order", "nan", "dimension", "integer", "target"])
def test_vector_input_guards(tmp_path, problem):
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    extras = {}
    if problem == "nan":
        query[0, 0] = np.nan
    if problem == "dimension":
        train = train[:, :4]
    if problem == "integer":
        query = query.astype(int)
    if problem == "target":
        extras["query_labels"] = np.zeros(5)
    path = tmp_path / "vectors.npz"
    write_vectors(path, fold, train, query, **extras)
    if problem == "order":
        fold["queries"] = fold["queries"][::-1]
    with pytest.raises(ValueError):
        s.load_vectors(path, fold, 8)


def test_normalization_large_and_zero_values():
    values = np.array([[1e300, 1e300], [0.0, 0.0], [1e-20, 0]])
    unit, zero = s.normalized_vectors(values, 1e-12)
    assert zero.tolist() == [False, True, True]
    assert np.isfinite(unit).all()
    assert np.linalg.norm(unit[0]) == pytest.approx(1)


def test_zero_query_stops():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    query[0] = 0
    with pytest.raises(ValueError, match="Zero-norm query"):
        s.prepare_selector(fold, train, query)


def test_zero_single_support_excluded_but_class_survives():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    train[0] = 0
    state = s.prepare_selector(fold, train, query)
    rows = s.select_batch(fold, state, 0, 5)
    assert all(r["semantic"]["positive"]["training_index"] != 0 for r in rows)
    assert state["zero_same_rule_supports"] == 1


def test_zero_whole_class_stops():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    train[[0, 3]] = 0
    with pytest.raises(ValueError, match="required class"):
        s.prepare_selector(fold, train, query)


def test_crossrule_rows_never_selected():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    query[:] = train[1]
    rows = s.select_batch(fold, s.prepare_selector(fold, train, query), 0, 5)
    assert all(
        r[m][k]["training_index"] != 1
        for r in rows
        for m in ("semantic", "lexical")
        for k in ("positive", "negative")
    )


def test_mapping_permutation_and_stable_exact_ties():
    fold = authored_plan()["folds"][0]
    train, query = np.ones((5, 8)), np.ones((5, 8))
    before = s.select_batch(fold, s.prepare_selector(fold, train, query), 0, 5)
    permutation = [4, 1, 3, 0, 2]
    other = copy.deepcopy(fold)
    other["training"] = [fold["training"][i] for i in permutation]
    after = s.select_batch(other, s.prepare_selector(other, train[permutation], query), 0, 5)
    for a, b in zip(before, after, strict=True):
        for method in ("semantic", "lexical"):
            for sign in ("positive", "negative"):
                assert a[method][sign]["support_id"] == b[method][sign]["support_id"]
    assert before[0]["semantic"]["positive"]["training_index"] == 3  # Beta, not matrix row 0.
    assert before[0]["semantic"]["negative"]["training_index"] == 2  # Alpha.


def test_lexical_matches_reviewed_legacy_algorithm():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    rows = s.select_batch(fold, s.prepare_selector(fold, train, query), 0, 5)
    pool = sorted(
        [r for r in fold["training"] if normalize(r["rule"]) == normalize(fold["rule"])],
        key=lambda r: (normalize(r["body"]), r["body"]),
    )
    labels = np.array([r["rule_violation"] for r in pool])
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    vectors = vectorizer.fit_transform([r["body"] for r in pool])
    scores = (vectorizer.transform([r["body"] for r in fold["queries"]]) @ vectors.T).toarray()
    for i, row in enumerate(rows):
        for label, sign in ((1, "positive"), (0, "negative")):
            indices = np.flatnonzero(labels == label)
            body = pool[indices[int(np.argmax(scores[i, indices]))]]["body"]
            assert fold["training"][row["lexical"][sign]["training_index"]]["body"] == body


def test_stable_selection_hash():
    fold = authored_plan()["folds"][0]
    train, query = vector_values(fold)
    state = s.prepare_selector(fold, train, query)
    assert s.fingerprint(s.select_batch(fold, state, 0, 5)) == s.fingerprint(
        s.select_batch(fold, state, 0, 5)
    )


def test_sample_group_deduplication_and_order_independence():
    fold = authored_plan()["folds"][0]
    fold["queries"].append({**fold["queries"][0], "row_id": 10})
    a = {normalize(fold["queries"][i]["body"]) for i in s.qualitative_sample(fold, 3, 2025)}
    fold["queries"].reverse()
    b = {normalize(fold["queries"][i]["body"]) for i in s.qualitative_sample(fold, 3, 2025)}
    assert a == b and len(a) == 3


def test_cues_are_diagnostics():
    value = s.cues('"You should not do that unless an exception applies."')
    assert value["quotation"] and value["advice_offer"] and value["negation"] and value["exception"]


def test_complete_replay_without_matrix_reads_or_selection(fixture, monkeypatch):
    first = run(fixture)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed computation was repeated")

    for name in ("load_vectors", "prepare_selector", "select_batch"):
        monkeypatch.setattr(s, name, forbidden)
    second = run(fixture)
    assert first == second
    replay = json.loads((fixture["output"] / "replay_check.json").read_text())
    assert replay["completed_result_reused"] is True
    assert first["official_metric"] is None and first["new_model_calls"] == 0


def test_return_bundle_contains_no_private_comments(fixture):
    summary = run(fixture)
    with zipfile.ZipFile(fixture["output"] / "return_bundle.zip") as archive:
        assert set(archive.namelist()) == {
            "audit.json",
            "replay_check.json",
            "public_contract.json",
        }
        text = b"".join(archive.read(name) for name in archive.namelist()).decode()
        for fold in fixture["plan"]["folds"]:
            for row in fold["training"] + fold["queries"]:
                assert row["body"] not in text
    assert summary["qualitative_review_completed"] is False
    assert summary["automatic_inference_allowed"] is False


def test_final_corruption_rejected(fixture):
    summary = run(fixture)
    path = fixture["output"] / summary["run_id"] / "private/selected_pairs.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="corruption"):
        run(fixture)


def test_input_corruption_rejected_before_selection(fixture, monkeypatch):
    path = fixture["inputs"] / "fold_0.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(s, "select_batch", lambda *a: pytest.fail("Unexpected computation"))
    with pytest.raises(ValueError, match="size"):
        run(fixture)


def test_interrupted_batches_reused(fixture, monkeypatch):
    actual = s.select_batch
    calls = []

    def interrupted(fold, state, start, end, tick):
        calls.append((fold["rule"], start))
        if len(calls) == 2:
            raise RuntimeError("Authored interruption")
        return actual(fold, state, start, end, tick)

    monkeypatch.setattr(s, "select_batch", interrupted)
    with pytest.raises(RuntimeError, match="Authored interruption"):
        run(fixture)
    assert len(list(fixture["output"].glob("*/fold_0/shards/*/complete.json"))) == 1
    resumed = []

    def guarded(fold, state, start, end, tick):
        assert not (fold["rule"] == fixture["plan"]["folds"][0]["rule"] and start == 0)
        resumed.append((fold["rule"], start))
        return actual(fold, state, start, end, tick)

    monkeypatch.setattr(s, "select_batch", guarded)
    result = run(fixture)
    assert result["total_queries"] == 10 and len(resumed) == 5


def test_manifest_path_traversal_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        rt.save_stage(tmp_path, "id", {"../bad.json": b"{}"})


def test_manifest_identity_mismatch(tmp_path):
    rt.save_stage(tmp_path, "correct", {"data.json": b"{}"})
    with pytest.raises(ValueError, match="contract"):
        rt.load_stage(tmp_path, "wrong")


def test_progress_timeout(tmp_path):
    with rt.Progress(tmp_path / "events.jsonl", 0.02, 0.005) as log:
        time.sleep(0.03)
        with pytest.raises(TimeoutError):
            log.check()
    records = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert any(r["event"] == "heartbeat" for r in records)
    assert all(r["utc"].endswith("+00:00") for r in records)


def test_parent_hard_time_cap(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_support_selection.py"),
            "--output",
            str(tmp_path),
            "--max-seconds",
            "0.001",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 124
    assert (tmp_path / "timeout.json").exists()


def test_recovery_offline_never_calls_aws(fixture, monkeypatch):
    monkeypatch.setattr(recovery, "aws_call", lambda *a, **kw: pytest.fail("Unexpected AWS call"))
    result = recovery.recover(fixture["config_path"], fixture["inputs"])
    assert result["status"] == "recovery_verified"
    assert result["source_tree_extracted"] is False
    assert result["prediction_arrays_read"] is False
    assert not (fixture["inputs"].parent.parent / "outside.py").exists()


def test_recovery_missing_inputs_no_implicit_download(fixture, tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "aws_call", lambda *a, **kw: pytest.fail("Unexpected AWS call"))
    with pytest.raises(FileNotFoundError):
        recovery.recover(fixture["config_path"], tmp_path / "empty")


@pytest.mark.parametrize("kind", ["duplicate", "link", "missing", "oversize"])
def test_plan_archive_safety(tmp_path, kind):
    path, target = tmp_path / "source.tar.gz", tmp_path / "plan.json"
    payload = b'{"safe":true}'
    make_archive(
        path,
        payload,
        member="other.json" if kind == "missing" else "adaptation_plan.json",
        duplicate=kind == "duplicate",
        link=kind == "link",
    )
    spec = {
        "member": "adaptation_plan.json",
        "max_bytes": 1 if kind == "oversize" else 100,
        "sha256": rt.sha_bytes(payload),
    }
    with pytest.raises(ValueError):
        recovery.extract_plan(path, target, spec)
    assert not target.exists()


def test_extract_does_not_overwrite_existing_different_plan(fixture):
    fixture["plan_path"].write_text("keep my changes")
    with pytest.raises(ValueError, match="Existing plan differs"):
        recovery.extract_plan(
            fixture["inputs"] / "source.tar.gz", fixture["plan_path"], fixture["config"]["plan"]
        )
    assert fixture["plan_path"].read_text() == "keep my changes"


def test_aws_account_mismatch_stops_before_download(fixture, tmp_path, monkeypatch):
    calls = []

    def wrong(args, *other, **kwargs):
        calls.append(args)
        return '{"Account":"000000000000"}'

    monkeypatch.setattr(recovery, "aws_call", wrong)
    with pytest.raises(ValueError, match="account differs"):
        recovery.recover(fixture["config_path"], tmp_path / "new", True, "test-profile")
    assert len(calls) == 1 and calls[0][0] == "sts"


def test_review_html_escapes_untrusted_text(fixture):
    plan = fixture["plan"]
    plan["folds"][0]["queries"][0]["body"] = '<script>alert("BAD")</script>'
    all_rows = []
    for fold in plan["folds"]:
        train, query = vector_values(fold)
        all_rows.append(s.select_batch(fold, s.prepare_selector(fold, train, query), 0, 5))
    page, key = audit.qualitative_html(plan, all_rows, [[0], [0]], "abcdef")
    assert b'<script>alert("BAD")' not in page
    assert b"&lt;script&gt;" in page
    assert len(key) == 2 and all(k["A"] != k["B"] for k in key)
    assert b"fetch(" not in page and b"XMLHttpRequest" not in page


def test_core_does_not_import_neural_libraries():
    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import jigsaw_rules.support_selection; "
            "assert not any(x in sys.modules for x in ('torch','transformers','peft')); print('passed')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def review_fixture(fixture):
    result = run(fixture)
    key = json.loads(
        (fixture["output"] / result["run_id"] / "private/qualitative_key.json").read_text()
    )
    ratings = {
        "schema": 1,
        "run_id": result["run_id"],
        "query_targets_read": False,
        "contains_comment_text": False,
        "ratings": [
            {
                "case_id": r["case_id"],
                "fold": r["fold"],
                "preference": "A" if r["A"] == "semantic" else "B",
                "issue": "none",
            }
            for r in key
        ],
    }
    path = fixture["inputs"] / "ratings.json"
    path.write_bytes(rt.json_bytes(ratings))
    return ratings, path


def test_review_decodes_blind_methods_and_is_repeatable(fixture):
    from scripts.summarize_support_selection_review import summarize

    _, path = review_fixture(fixture)
    result = summarize(fixture["output"], path)
    assert result["cases"] == 6
    assert all(p["preferences"] == {"semantic": 3} for p in result["policies"].values())
    assert result == summarize(fixture["output"], path)
    assert result["automatic_inference_allowed"] is False


@pytest.mark.parametrize("failure", ["incomplete", "duplicate", "target", "wrong_run"])
def test_review_rejects_invalid_ratings(fixture, failure):
    from scripts.summarize_support_selection_review import summarize

    ratings, path = review_fixture(fixture)
    if failure == "incomplete":
        ratings["ratings"].pop()
    if failure == "duplicate":
        ratings["ratings"][-1] = ratings["ratings"][0]
    if failure == "target":
        ratings["ratings"][0]["rule_violation"] = 1
    if failure == "wrong_run":
        ratings["run_id"] = "0" * 24
    path.write_bytes(rt.json_bytes(ratings))
    with pytest.raises(ValueError):
        summarize(fixture["output"], path)


def test_changed_review_cannot_replace_frozen_review(fixture):
    from scripts.summarize_support_selection_review import summarize

    ratings, path = review_fixture(fixture)
    summarize(fixture["output"], path)
    ratings["ratings"][0]["preference"] = "tie"
    path.write_bytes(rt.json_bytes(ratings))
    with pytest.raises(ValueError, match="contract mismatch"):
        summarize(fixture["output"], path)


@pytest.fixture
def install_fixture(tmp_path, monkeypatch):
    import importlib.util

    # Installer is distributed beside the ZIP README, not in the project import path.
    path = Path(os.environ.get("JIGSAW_BUNDLE_ROOT", str(ROOT))) / "apply_bundle.py"
    if not path.exists():
        pytest.skip("Installer contract tests run from the distributed bundle only")
    spec = importlib.util.spec_from_file_location("jigsaw_patch_installer", path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    repo, bundle = tmp_path / "repo", tmp_path / "bundle"
    (repo / "docs").mkdir(parents=True)
    bundle.mkdir()
    (repo / installer.DOC).write_text("# Existing recorded history\n")
    for args in (
        ("init",),
        ("config", "user.email", "fixture@example.invalid"),
        ("config", "user.name", "Authored fixture"),
        ("add", "docs"),
        ("commit", "-m", "Authored baseline"),
    ):
        installer.git(repo, *args)
    base = installer.git(repo, "rev-parse", "HEAD").decode().strip()
    monkeypatch.setattr(installer, "EXPECTED_BASE", base)
    monkeypatch.setattr(
        installer,
        "DOC_BLOB",
        installer.git(repo, "rev-parse", "HEAD:" + installer.DOC).decode().strip(),
    )
    files = {"src/new.py": b'print("authored")\n', "append.md": b"## Pending audit\n"}
    for name, value in files.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    manifest = {
        "schema": 1,
        "base_commit": base,
        "install_files": ["src/new.py"],
        "documentation_append": "append.md",
        "files": {k: {"bytes": len(v), "sha256": installer.sha(v)} for k, v in files.items()},
    }
    (bundle / "SHA256SUMS.json").write_text(json.dumps(manifest))
    return installer, repo, bundle


def test_installer_additive_and_idempotent(install_fixture):
    installer, repo, bundle = install_fixture
    first = installer.install(bundle, repo)
    second = installer.install(bundle, repo)
    assert first["writes_this_invocation"] == 2
    assert second["writes_this_invocation"] == 0
    assert (repo / installer.DOC).read_text().startswith("# Existing recorded history\n")
    assert (repo / installer.DOC).read_text().count("## Pending audit") == 1


def test_installer_conflict_leaves_all_files_unchanged(install_fixture):
    installer, repo, bundle = install_fixture
    (repo / "src").mkdir()
    (repo / "src/new.py").write_text("user work")
    before = (repo / installer.DOC).read_bytes()
    with pytest.raises(ValueError, match="destination already differs"):
        installer.install(bundle, repo)
    assert (repo / "src/new.py").read_text() == "user work"
    assert (repo / installer.DOC).read_bytes() == before


def test_installer_checksum_failure_has_no_side_effects(install_fixture):
    installer, repo, bundle = install_fixture
    (bundle / "src/new.py").write_text("changed bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        installer.install(bundle, repo)
    assert not (repo / "src/new.py").exists()


def test_installer_rejects_changed_head(install_fixture):
    installer, repo, bundle = install_fixture
    installer.git(repo, "commit", "--allow-empty", "-m", "Different head")
    with pytest.raises(ValueError, match="HEAD changed"):
        installer.install(bundle, repo)
    assert not (repo / "src/new.py").exists()
