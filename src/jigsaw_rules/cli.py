"""Project entry point. No secrets are printed or stored in project files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jigsaw_rules.cloud import backup, restore
from jigsaw_rules.data import audit, load_data, synthetic
from jigsaw_rules.download import download
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.review import review_run
from jigsaw_rules.runtime import Progress, atomic_json, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "download",
            "audit",
            "baseline",
            "smoke",
            "backup",
            "restore",
            "review",
            "semantic",
            "model-download",
            "features",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--run-id", help="Review an existing run without retraining")
    parser.add_argument(
        "--allow-synthetic", action="store_true", help="Allow a labeled demo review"
    )
    parser.add_argument(
        "--cloud", action="store_true", help="Back up after every completed experiment stage"
    )
    parser.add_argument("--baseline-run", default="c15c2c2318fc0ed619c6")
    parser.add_argument(
        "--export", action="store_true", help="Publish reviewed aggregate feature evidence"
    )
    args = parser.parse_args()
    if args.export and args.command != "features":
        parser.error("--export is only available for the feature experiment")
    root = args.root.resolve()
    config = None
    if args.cloud or args.command in {"backup", "restore"}:
        config_path = root / "configs/local.json"
        if not config_path.exists():
            raise FileNotFoundError("Create configs/local.json using configs/local.example.json")
        config = json.loads(config_path.read_text())
    with Progress(root / "logs/commands.jsonl", args.command):
        if args.command == "preflight":
            record = environment()
            record["project_root"] = str(root)
            record["data_ready"] = all(
                (root / "data/raw" / n).exists()
                for n in ["train.csv", "test.csv", "sample_submission.csv"]
            )
            atomic_json(root / "logs/preflight.json", record)
            print(json.dumps(record, indent=2))
        elif args.command == "download":
            download(root / "data/raw")
        elif args.command == "audit":
            train, test, _ = load_data(root / "data/raw")
            record = audit(train, test)
            atomic_json(root / "runs/data_audit.json", record)
            print(json.dumps(record, indent=2))
        elif args.command == "baseline":
            run_baseline(root, folds=args.folds, cloud=config)
        elif args.command == "smoke":
            sample_dir = root / "data/synthetic"
            if not (sample_dir / "SYNTHETIC.txt").exists():
                synthetic(sample_dir)
            run_baseline(root, data_dir=sample_dir)
        elif args.command in {"semantic", "model-download"}:
            from jigsaw_rules.embeddings import load_spec, prepare_model
            from jigsaw_rules.semantic_pipeline import run_semantic

            spec = load_spec(root)
            if args.command == "semantic":
                run_semantic(root, spec, cloud=config)
            else:
                prepare_model(root, spec)
        elif args.command == "features":
            from jigsaw_rules.features import export_features, run_features

            directory = run_features(root, args.baseline_run, cloud=config)
            if args.export:
                export_features(root, directory)
                print("FEATURE_AGGREGATES_EXPORTED")
        elif args.command == "backup":
            backup(root, config["bucket"], config["region"])
        elif args.command == "review":
            review_run(root, args.run_id, allow_synthetic=args.allow_synthetic)
        else:
            restore(root, config["bucket"], config["region"])


if __name__ == "__main__":
    main()
