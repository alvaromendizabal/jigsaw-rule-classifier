"""Pinned post-competition data with a target-blind research/confirmation boundary."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from jigsaw_rules.data import EXAMPLES, TEXT, normalize, validate_frame
from jigsaw_rules.runtime import Progress, atomic_bytes, atomic_json, digest, stage

PROTOCOL_COMMIT = "9b4d66448998da818b71e4e697aa8e937c9222f0"
EXPOSURE = ["body", *EXAMPLES]
PUBLIC_FILES = {"source.json", "boundary.json", "metric.json"}
ROLES = {"research", "confirmation", "historical_overlap", "boundary_overlap"}


def key(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_plan(root: Path) -> dict:
    plan = json.loads((root / "configs/released.json").read_text())
    research, reserved = set(plan["research_rule_ids"]), set(plan["reserved_rule_ids"])
    if (
        plan["schema"] != 1
        or research & reserved
        or research | reserved != set(plan["rule_text"])
        or plan["research_usage"] != "Public"
        or plan["reserved_usage"] != "Private"
        or plan["historical_exposure"] != EXPOSURE
    ):
        raise ValueError("Unsupported released-data boundary contract")
    return plan


def verify_source(folder: Path, plan: dict) -> None:
    for name, record in plan["source"]["files"].items():
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("Unsafe release member")
        path = folder / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]
        ):
            raise ValueError(f"Released source checksum mismatch: {name}")


def download_release(root: Path, plan: dict) -> Path:
    """Public, version-pinned download; existing files are verified before reuse."""
    folder = root / "data/released/v1"
    folder.mkdir(parents=True, exist_ok=True)
    source = plan["source"]
    with FileLock(str(folder / "download.lock"), timeout=1):
        if all((folder / name).exists() for name in source["files"]):
            verify_source(folder, plan)
            return folder
        archive = folder / "release.zip"
        if not archive.exists():
            partial = folder / "release.zip.partial"
            with (
                Progress(root / "logs/released.jsonl", "released_download"),
                urllib.request.urlopen(source["source_url"], timeout=60) as response,
                partial.open("wb") as output,
            ):
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > source["archive_bytes"]:
                        raise ValueError("Released archive exceeds its pinned byte count")
                    output.write(block)
            if digest(partial) != source["archive_sha256"]:
                raise ValueError("Released archive checksum mismatch")
            partial.replace(archive)
        if digest(archive) != source["archive_sha256"]:
            raise ValueError("Released archive checksum mismatch")
        with zipfile.ZipFile(archive) as stream:
            if set(stream.namelist()) != set(source["files"]) or len(stream.namelist()) != len(
                source["files"]
            ):
                raise ValueError("Unexpected release archive members")
            for name, record in source["files"].items():
                if Path(name).name != name or stream.getinfo(name).file_size != record["bytes"]:
                    raise ValueError("Unsafe or oversized release archive member")
                payload = stream.read(name)
                if hashlib.sha256(payload).hexdigest() != record["sha256"]:
                    raise ValueError("Released member checksum mismatch")
                path = folder / name
                if path.exists() and digest(path) != record["sha256"]:
                    raise ValueError("Refusing to replace a modified released source")
                atomic_bytes(path, payload)
        verify_source(folder, plan)
    return folder


def partition_release(
    historical: pd.DataFrame, inputs: pd.DataFrame, metadata: pd.DataFrame, plan: dict
) -> tuple[pd.DataFrame, dict]:
    """Assign roles using metadata and text only; targets are forbidden inputs."""
    validate_frame(inputs, train=False)
    if set(metadata.columns) != {"row_id", "rule_id", "Usage"}:
        raise ValueError("Partition metadata must exclude target values")
    if (
        metadata.row_id.isna().any()
        or metadata.row_id.duplicated().any()
        or set(metadata.row_id) != set(inputs.row_id)
        or set(metadata.Usage) - {"Public", "Private"}
        or set(metadata.rule_id) != set(plan["rule_text"])
    ):
        raise ValueError("Release metadata identity, Usage or policy coverage differs")
    frame = inputs.merge(metadata, on="row_id", validate="one_to_one", sort=False)
    if not frame.rule.eq(frame.rule_id.map(plan["rule_text"])).all():
        raise ValueError("Released rule text and solution policy ID disagree")
    for column in EXPOSURE:
        if not historical[column].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError("Historical exposure contains missing text")
    prior = set(historical[EXPOSURE].map(normalize).to_numpy().ravel())
    texts = frame[EXPOSURE].map(normalize)
    reserved = frame.Usage.eq("Private") | frame.rule_id.isin(plan["reserved_rule_ids"])
    known = texts.body.isin(prior)
    protected = reserved & ~known
    protected_bodies = set(texts.loc[protected, "body"])
    crosses = texts.isin(protected_bodies).any(axis=1)
    research = ~reserved & ~crosses
    roles = np.select(
        [protected, reserved & known, ~reserved & crosses, research],
        ["confirmation", "historical_overlap", "boundary_overlap", "research"],
        default="unassigned",
    )
    if "unassigned" in roles or not research.any() or not protected.any():
        raise ValueError("Boundary produced an empty or unassigned partition")
    research_texts = set(texts.loc[research].to_numpy().ravel()) | prior
    if research_texts & protected_bodies:
        raise ValueError("Research context exposes a reserved comment")
    assignments = frame[["row_id", "rule_id", "Usage"]].copy()
    assignments["role"] = roles
    assignments["body_sha256"] = texts.body.map(
        lambda text: hashlib.sha256(text.encode()).hexdigest()
    )
    assignments = assignments.sort_values("row_id").reset_index(drop=True)
    counts = assignments.groupby(["rule_id", "Usage", "role"]).size().reset_index(name="rows")
    report = {
        "rows": len(frame),
        "rule_count": len(plan["rule_text"]),
        "counts": counts.to_dict("records"),
        "role_counts": assignments.role.value_counts().sort_index().to_dict(),
        "usage_counts": assignments.Usage.value_counts().sort_index().to_dict(),
        "historically_exposed_unique_texts": len(prior),
        "research_context_confirmation_body_overlap": 0,
        "reserved_policy_ids": plan["reserved_rule_ids"],
        "research_policy_ids": plan["research_rule_ids"],
        "confirmation_targets_accessed": False,
        "confirmation_predictions_generated": False,
        "assignment_uses_targets": False,
        "approximate_copy_audit": "pending on the new boundary",
        "protocol_commit": PROTOCOL_COMMIT,
    }
    return assignments, report


def read_research_targets(path: Path, row_ids: set[int], research_rules: set[str]) -> pd.DataFrame:
    """Skip reserved rows before interpreting their target field; no holdout mode."""
    result, seen = [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["row_id", "Usage", "rule_violation", "rule"]:
            raise ValueError("Unexpected solution schema")
        for row in reader:
            row_id = int(row["row_id"])
            if row_id in seen:
                raise ValueError("Duplicate solution row ID")
            seen.add(row_id)
            if row_id not in row_ids:
                continue
            if (
                row["Usage"] != "Public"
                or row["rule"] not in research_rules
                or row["rule_violation"] not in {"0", "1"}
            ):
                raise ValueError("Research targets must be public and binary")
            result.append({"row_id": row_id, "rule_violation": int(row["rule_violation"])})
    if {row["row_id"] for row in result} != row_ids:
        raise ValueError("Missing research target IDs")
    return pd.DataFrame(result).sort_values("row_id").reset_index(drop=True)


def metric_attachment_audit(frame: pd.DataFrame, columns: list[str], tolerance: float) -> dict:
    """Check organizer score arithmetic; report discrepancies instead of hiding them."""
    values = frame[["PrivateScore", *columns]].apply(pd.to_numeric, errors="raise")
    complete = np.isfinite(values.to_numpy()).all(axis=1)
    difference = (
        values.loc[complete, columns].mean(axis=1) - values.loc[complete, "PrivateScore"]
    ).abs()
    return {
        "rows": len(frame),
        "complete_rows": int(complete.sum()),
        "incomplete_rows": int((~complete).sum()),
        "absolute_tolerance": tolerance,
        "matching_rows": int(difference.le(tolerance).sum()),
        "discrepant_complete_rows": int(difference.gt(tolerance).sum()),
        "maximum_absolute_difference": float(difference.max()),
        "median_absolute_difference": float(difference.median()),
        "interpretation": "Organizer per-policy scores strongly corroborate equal-weight rule "
        "macro AUC. Discrepant/incomplete rows remain unresolved; executable scorer parity "
        "and a project leaderboard result are not established.",
    }


def _implementation() -> dict:
    return {"released.py": digest(Path(__file__)), "normalize": key(inspect.getsource(normalize))}


def prepare_release(root: Path) -> Path:
    plan = load_plan(root)
    folder = download_release(root, plan)
    # Identity checks occur before any released target is interpreted.
    original = root / "data/raw"
    for old, new in (("train.csv", "train.csv"), ("test.csv", "public_test.csv")):
        if digest(original / old) != plan["source"]["files"][new]["sha256"]:
            raise ValueError("Original data differs from the locked historical exposure")
    contract = {
        "schema": 1,
        "plan": plan,
        "implementation": _implementation(),
        "historical": {name: digest(original / name) for name in ("train.csv", "test.csv")},
        "protocol_commit": PROTOCOL_COMMIT,
    }
    directory = root / "runs/released" / key(contract)[:20]

    def action(work: Path) -> None:
        inputs = pd.read_csv(folder / "test.csv")
        metadata = pd.read_csv(folder / "solution.csv", usecols=["row_id", "Usage", "rule"])
        metadata = metadata.rename(columns={"rule": "rule_id"})
        training = pd.read_csv(original / "train.csv")
        historical = pd.concat([training[TEXT], pd.read_csv(original / "test.csv")[TEXT]])
        assignments, report = partition_release(historical, inputs, metadata, plan)
        ids = set(assignments.loc[assignments.role.eq("research"), "row_id"])
        targets = read_research_targets(
            folder / "solution.csv", ids, set(plan["research_rule_ids"])
        )
        research = inputs.loc[inputs.row_id.isin(ids)].merge(
            targets, on="row_id", validate="one_to_one"
        )
        validate_frame(research, train=True)
        report["original_training_rows"] = len(training)
        report["combined_research_rows_before_deduplication"] = len(training) + len(research)
        combined = pd.concat([training, research], ignore_index=True)
        pair_groups = (
            combined.assign(body_key=combined.body.map(normalize))
            .groupby(["rule", "body_key"])
            .rule_violation
        )
        report["combined_repeated_body_policy_rows"] = int(pair_groups.size().sub(1).sum())
        report["combined_conflicting_body_policy_groups"] = int(pair_groups.nunique().gt(1).sum())
        conflicting = pair_groups.nunique().loc[lambda values: values.gt(1)].index
        subreddit_counts = (
            combined.assign(body_key=combined.body.map(normalize))
            .groupby(["rule", "body_key"])
            .subreddit.nunique()
            .reindex(conflicting)
        )
        report["conflicting_groups_with_same_subreddit"] = int(subreddit_counts.eq(1).sum())
        report["conflicting_groups_with_multiple_subreddits"] = int(subreddit_counts.gt(1).sum())
        report["research_label_counts"] = (
            research.merge(assignments[["row_id", "rule_id"]], on="row_id", validate="one_to_one")
            .groupby(["rule_id", "rule_violation"])
            .size()
            .reset_index(name="rows")
            .to_dict("records")
        )
        assignments.to_csv(work / "assignments.csv", index=False)
        research.sort_values("row_id").to_csv(work / "research.csv", index=False)
        atomic_json(work / "contract.json", contract)
        atomic_json(work / "boundary.json", report)

    return stage(directory, "boundary", action)


def load_research(root: Path) -> pd.DataFrame:
    """Return only the current verified research export, never the solution file."""
    evidence = released_evidence(root)
    if evidence is None:
        raise ValueError("Prepare and export the released-data boundary first")
    directory = root / "runs/released" / evidence["metadata"]["run_id"] / "boundary"
    marker = directory / "complete.json"
    if digest(marker) != evidence["metadata"]["private_checkpoint_sha256"]:
        raise ValueError("Released boundary marker differs")
    records = json.loads(marker.read_text())["files"]
    if set(records) != {"assignments.csv", "research.csv", "contract.json", "boundary.json"}:
        raise ValueError("Unexpected released checkpoint members")
    for name, sha in records.items():
        if digest(directory / name) != sha:
            raise ValueError("Released research checkpoint checksum mismatch")
    assignments = pd.read_csv(directory / "assignments.csv")
    permitted = assignments.loc[assignments.role.eq("research")]
    if (
        not permitted.Usage.eq("Public").all()
        or not permitted.rule_id.isin(evidence["boundary"]["research_policy_ids"]).all()
    ):
        raise ValueError("Reserved rows entered the research assignment")
    research = pd.read_csv(directory / "research.csv")
    validate_frame(research, train=True)
    if set(research.row_id) != set(permitted.row_id):
        raise ValueError("Research export does not match permitted row IDs")
    return research


def export_release(root: Path, directory: Path) -> None:
    """Publish aggregates only; verify source, assignments and stage hashes first."""
    plan = load_plan(root)
    verify_source(root / "data/released/v1", plan)
    marker = json.loads((directory / "complete.json").read_text())
    expected = {"assignments.csv", "research.csv", "contract.json", "boundary.json"}
    if set(marker["files"]) != expected or any(
        digest(directory / name) != sha for name, sha in marker["files"].items()
    ):
        raise ValueError("Released boundary checkpoint checksum mismatch")
    contract = json.loads((directory / "contract.json").read_text())
    if (
        contract["implementation"] != _implementation()
        or contract["plan"] != plan
        or directory.parent.name != key(contract)[:20]
        or contract["historical"]
        != {name: digest(root / "data/raw" / name) for name in ("train.csv", "test.csv")}
    ):
        raise ValueError("Released boundary implementation, plan or historical inputs changed")
    official = plan["official_metric_evidence"]
    attachment = root / "reports/private/official-rule-auc.csv"
    if not attachment.exists():
        with urllib.request.urlopen(official["url"], timeout=60) as response:
            payload = response.read(2_000_000)
        if hashlib.sha256(payload).hexdigest() != official["sha256"]:
            raise ValueError("Organizer metric attachment checksum mismatch")
        atomic_bytes(attachment, payload)
    if digest(attachment) != official["sha256"]:
        raise ValueError("Organizer metric attachment checksum mismatch")
    metric = metric_attachment_audit(
        pd.read_csv(attachment), official["rule_columns"], official["absolute_tolerance"]
    )
    metric.update({"source_url": official["url"], "source_sha256": official["sha256"]})
    public = root / "reports/released"
    atomic_json(public / "source.json", plan["source"])
    atomic_bytes(public / "boundary.json", (directory / "boundary.json").read_bytes())
    atomic_json(public / "metric.json", metric)
    atomic_json(
        public / "metadata.json",
        {
            "schema": 1,
            "run_id": directory.parent.name,
            "plan_sha256": digest(root / "configs/released.json"),
            "implementation": _implementation(),
            "private_checkpoint_sha256": digest(directory / "complete.json"),
            "files": {name: digest(public / name) for name in sorted(PUBLIC_FILES)},
        },
    )


def released_evidence(root: Path) -> dict | None:
    folder = root / "reports/released"
    if not (folder / "metadata.json").exists():
        return None
    metadata = json.loads((folder / "metadata.json").read_text())
    if (
        metadata["schema"] != 1
        or set(metadata["files"]) != PUBLIC_FILES
        or metadata["plan_sha256"] != digest(root / "configs/released.json")
        or metadata["implementation"] != _implementation()
    ):
        raise ValueError("Released evidence source contract differs")
    for name, sha in metadata["files"].items():
        if (folder / name).is_symlink() or digest(folder / name) != sha:
            raise ValueError("Released public evidence checksum mismatch")
    result = {
        name.removesuffix(".json"): json.loads((folder / name).read_text()) for name in PUBLIC_FILES
    }
    boundary = result["boundary"]
    if (
        set(boundary["role_counts"]) - ROLES
        or sum(boundary["role_counts"].values()) != boundary["rows"]
        or boundary["research_context_confirmation_body_overlap"] != 0
        or boundary["confirmation_targets_accessed"]
        or boundary["confirmation_predictions_generated"]
        or boundary["assignment_uses_targets"]
    ):
        raise ValueError("Released confirmation boundary is not intact")
    return {**result, "metadata": metadata}
