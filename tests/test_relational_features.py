import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import relational_features as rf
from scripts import run_relational_features as run


def values(text):
    names, vals = rf.relation_row(text)
    return dict(zip(names, vals, strict=True))


@pytest.mark.parametrize(
    "text,feature",
    [
        ("Should I sue my landlord?", "act_roles/self_request_legal_present"),
        ("Could you help with my court case?", "act_roles/other_request_legal_present"),
        ("You should sue your landlord.", "act_roles/directive_legal_present"),
        ("I can help with the legal appeal.", "act_roles/offer_legal_present"),
        ("I had a lawyer for my case.", "act_roles/experience_legal_present"),
        ("According to the court report, it ended.", "act_roles/reported_legal_present"),
        ("Please buy at https://shop.example.org/item", "link_intent/cta_url_present"),
        ("Read the source https://example.org/info", "link_intent/resource_url_present"),
        ("My shop is https://shop.example.org", "link_intent/owned_business_url_present"),
        ("https://example.org/?ref=author", "link_intent/referral_parameter_present"),
        ("Contact me for a discount.", "link_intent/contact_price_present"),
        ("You should not sue.", "action_scope/directive_legal_negated_present"),
        ("Do not buy at this site.", "action_scope/cta_negated_present"),
        ('He said "You should sue".', "action_scope/quoted_directive_legal_present"),
        (
            "I am not a lawyer. You should sue.",
            "action_scope/disclaimer_then_directive_legal_present",
        ),
    ],
)
def test_expected_authored_behavior(text, feature):
    assert values(text)[feature] == 1


@pytest.mark.parametrize("text", ['"You should sue"', "“You should sue”", "> You should sue\n"])
def test_quoted_directive_not_authored(text):
    v = values(text)
    assert v["act_roles/directive_legal_present"] == 0
    assert v["action_scope/quoted_directive_legal_present"] == 1


def test_code_excluded():
    assert values("`You should sue` discussion")["act_roles/directive_legal_present"] == 0


def test_clause_boundary_prevents_topic_crossing():
    assert (
        values("You should visit. Legal matters are different.")[
            "act_roles/directive_legal_present"
        ]
        == 0
    )


def test_quotation_gap_never_joins_context():
    assert values('You should "buy this" sue.')["act_roles/directive_legal_present"] == 0


def test_url_sentence_boundary_not_ignored():
    assert values("A source https://example.org. Buy now.")["link_intent/cta_url_present"] == 0


def test_negated_directive_not_positive():
    v = values("You should not sue.")
    assert v["action_scope/directive_legal_positive_present"] == 0
    assert v["action_scope/directive_legal_negated_present"] == 1


def test_disclaimer_does_not_negate_next_directive():
    v = values("Not a lawyer. You should sue.")
    assert v["action_scope/directive_legal_positive_present"] == 1


@pytest.mark.parametrize("bad", [None, 1, "", "   ", np.nan])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        rf.relation_row(bad)


def test_does_not_use_targets_or_other_columns():
    a = pd.DataFrame({"body": ["You should sue", "Buy now"], "rule_violation": [0, 1]})
    b = a.assign(rule_violation=[1, 0], rule=["anything", "anything"])
    pd.testing.assert_frame_equal(rf.relational_features(a), rf.relational_features(b))


def test_feature_schema_and_repeatability():
    a = rf.relational_features(pd.DataFrame({"body": ["Visit my shop", "Could I sue?"]}))
    assert a.shape == (2, 72) and np.isfinite(a.to_numpy()).all()
    assert set(a) == {c for group in rf.CATALOG.values() for c in group}


def test_case_insensitive():
    assert values("you should sue") == values("YOU SHOULD SUE")


def test_local_window_no_far_binding():
    v = values("You should " + "word " * 30 + "sue")
    assert v["act_roles/directive_legal_present"] == 0


def test_cached_reference_order_and_parity(tmp_path):
    p = tmp_path / "p.npz"
    np.savez(
        p,
        row_ids=np.array([1, 2]),
        probability=np.array([0.5, 0.5]),
        coefficients=np.zeros((1, 3)),
        intercept=np.array([0.0]),
    )
    np.testing.assert_array_equal(
        run.load_reference(p, np.array([1, 2]), np.zeros((2, 3))), [0.5, 0.5]
    )
    with pytest.raises(ValueError, match="order"):
        run.load_reference(p, np.array([2, 1]))
    np.savez(
        p,
        row_ids=np.array([1, 2]),
        probability=np.array([0.9, 0.9]),
        coefficients=np.zeros((1, 3)),
        intercept=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="parity"):
        run.load_reference(p, np.array([1, 2]), np.zeros((2, 3)))


def test_stage_corruption_stops(tmp_path):
    run.stage_write(tmp_path, {"data.txt": b"valid"}, {"a": 1})
    (tmp_path / "data.txt").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="checksum"):
        run.stage_read(tmp_path, {"a": 1})


def test_stage_missing_marker_is_not_completed(tmp_path):
    (tmp_path / "data.txt").write_bytes(b"partial")
    assert run.stage_read(tmp_path, {"a": 1}) is None


def test_no_gpu_authorization_and_predeclared_fit_count():
    assert len(run.NEW) == 9
    assert len(run.VARIANTS) == 11
    config = json.loads((Path(__file__).parents[1] / run.CONFIG).read_text())
    assert config["primary"] == "add_act_roles vs add_behavior"
    assert config["automatic_gpu_authorization"] is False
    assert config["new_fits"] == 18


def test_train_only_screen_query_does_not_change_selection():
    t = pd.DataFrame({"one": [0.0, 0.0, 1.0, 1.0], "two": [1.0, 0.0, 1.0, 0.0]})
    a = run.transform_block(t, t, np.array([0, 0, 1, 1]), list(t), 1)
    b = run.transform_block(t, t * 1000, np.array([0, 0, 1, 1]), list(t), 1)
    assert a[2] == b[2]


def test_atomic_write_and_hash_stable(tmp_path):
    run.atomic_json(tmp_path / "a.json", {"key": 1})
    before = hashlib.sha256((tmp_path / "a.json").read_bytes()).hexdigest()
    run.atomic_json(tmp_path / "a.json", {"key": 1})
    assert run.digest(tmp_path / "a.json") == before


@pytest.fixture(scope="module")
def study_template(tmp_path_factory):
    import shutil

    from scripts import run_behavioral_features as old
    from tests.test_behavioral_features import authored_frame

    root = tmp_path_factory.mktemp("round2_integration")
    source = Path(__file__).parents[1]
    for name in set(run.SOURCE_PATHS):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    (root / "data/raw").mkdir(parents=True)
    frame = authored_frame()
    frame.to_csv(root / "data/raw/train.csv", index=False)
    spec = json.loads((root / old.CONFIG).read_text())
    spec.update(
        word_features=120,
        char_features=150,
        feature_budget_per_family=8,
        bootstrap_replicates=50,
        max_seconds=120,
    )
    prior = old.run_study(root, test_frame=frame, test_config=spec)
    config = json.loads((root / run.CONFIG).read_text())
    config.update(
        prior_run_id=prior["run_id"],
        prior_results_sha256=run.digest(root / ("reports/behavioral_features/results.json")),
        train_sha256=run.digest(root / "data/raw/train.csv"),
        query_counts=[48, 48],
        bootstrap_replicates=50,
        max_seconds=120,
    )
    (root / run.CONFIG).write_text(json.dumps(config, indent=2) + "\n")
    return root


@pytest.fixture
def study_root(study_template, tmp_path):
    import shutil

    return Path(shutil.copytree(study_template, tmp_path / "project"))


def test_full_study_reuses_four_old_controls(study_root):
    before = {
        str(p): run.digest(p)
        for path in ("runs/behavioral_features", "reports/behavioral_features")
        for p in (study_root / path).rglob("*")
        if p.is_file()
    }
    result = run.run_study(study_root)
    assert result["new_fits"] == 18 and result["reused_round1_controls"] == 4
    assert result["automatic_gpu_authorization"] is False
    assert all(r["design_parity"] for r in result["control_parity"])
    assert all(run.digest(Path(name)) == digest for name, digest in before.items())
    assert len(run.figures(result)) == 8


def test_interruption_reuses_completed_candidates(study_root):
    with pytest.raises(TimeoutError, match="authored interruption"):
        run.run_study(study_root, max_new_fits=3)
    result = run.run_study(study_root)
    assert result["new_fits"] == 15 and result["reused_new_fits"] == 3


def test_complete_replay_never_fits(study_root, monkeypatch):
    result = run.run_study(study_root)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit called on completed replay")

    monkeypatch.setattr(run, "fit_candidate", forbidden)
    assert run.run_study(study_root) == result


def test_reference_source_changed_stops(study_root):
    (study_root / "scripts/behavioral_features.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        run.run_study(study_root)


def test_reference_checkpoint_missing_stops(study_root):
    config = json.loads((study_root / run.CONFIG).read_text())
    path = study_root / "runs/behavioral_features" / config["prior_run_id"]
    (path / "fold_0/add_behavior/complete.json").unlink()
    with pytest.raises(ValueError, match="checkpoint missing"):
        run.run_study(study_root)


def test_completed_corruption_rejected(study_root):
    result = run.run_study(study_root)
    path = study_root / run.PRIVATE / result["run_id"] / "fold_0/add_act_roles/predictions.npz"
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="checksum"):
        run.run_study(study_root)


def test_public_export_excludes_raw_and_predictions(study_root, tmp_path):
    import zipfile

    run.run_study(study_root)
    path = run.export_return(study_root, tmp_path / "out.zip")
    with zipfile.ZipFile(path) as z:
        assert not any(name.startswith("data/") or name.endswith(".npz") for name in z.namelist())
        hashes = json.loads(z.read("SHA256SUMS.json"))
        for name, expected in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
