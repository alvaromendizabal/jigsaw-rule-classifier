"""Project entry point. No secrets are printed or stored in project files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jigsaw_rules.cloud import backup, restore
from jigsaw_rules.data import audit, load_data, synthetic
from jigsaw_rules.download import download
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.runtime import Progress, atomic_json, environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "download", "audit", "baseline", "smoke", "backup", "restore"],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--cloud", action="store_true", help="Back up after every completed baseline stage"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = None
    if args.cloud or args.command in {"backup", "restore"}:
        config_path = root / "configs/local.json"
        if not config_path.exists():
            raise FileNotFoundError("Run bootstrap.sh to install the local AWS configuration")
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
        elif args.command == "backup":
            backup(root, config["bucket"], config["region"])
        else:
            restore(root, config["bucket"], config["region"])


if __name__ == "__main__":
    main()
