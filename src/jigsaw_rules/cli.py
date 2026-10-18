"""Run the maintained CPU reference; synthetic data are explicitly labelled."""

from __future__ import annotations

import argparse
from pathlib import Path

from jigsaw_rules.data import synthetic
from jigsaw_rules.pipeline import run_baseline
from jigsaw_rules.review import review_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "baseline", "review"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "review":
        review_run(root, args.run_id, allow_synthetic=args.allow_synthetic)
    elif args.command == "demo":
        data = root / "data/synthetic"
        if data.exists() and any(data.iterdir()) and not (data / "SYNTHETIC.txt").is_file():
            raise ValueError("Refusing to overwrite an unmarked input directory")
        if not (data / "SYNTHETIC.txt").is_file():
            synthetic(data)
        run_baseline(root, data_dir=data)
    else:
        run_baseline(root)


if __name__ == "__main__":
    main()
