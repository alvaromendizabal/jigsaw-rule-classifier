"""Package the accepted route or run verified offline benchmark inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from jigsaw_rules.offline import package_model, predict_file
from jigsaw_rules.runtime import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package")
    package.add_argument("--destination", type=Path, default=Path("runs/delivery"))
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--manifest-sha256", required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, default=Path("runs/offline_output"))
    predict.add_argument("--cache", type=Path, default=Path("runs/offline_cache"))
    predict.add_argument("--batch-rows", type=int, default=32)
    args = parser.parse_args()
    if args.command == "package":
        root = Path(__file__).resolve().parents[1]
        path = package_model(root, args.destination)
        print(f"Bundle: {path}\nManifest SHA-256: {digest(path / 'bundle.json')}")
    else:
        path = predict_file(
            args.bundle,
            args.manifest_sha256,
            args.input,
            args.output,
            args.cache,
            batch_rows=args.batch_rows,
        )
        print(f"Verified benchmark predictions: {path}")


if __name__ == "__main__":
    main()
