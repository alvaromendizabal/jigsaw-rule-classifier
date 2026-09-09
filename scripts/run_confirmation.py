"""Prepare target-free confirmation inputs or replay fixed inference."""

import argparse
from pathlib import Path

from jigsaw_rules.confirmation import prepare_inputs, run_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "predict"])
    parser.add_argument("--run-id")
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        print(prepare_inputs(root))
        return
    if (
        not args.run_id
        or len(args.run_id) != 20
        or any(c not in "0123456789abcdef" for c in args.run_id)
    ):
        raise ValueError("An exact confirmation run identifier is required")
    print(run_predictions(root, root / "runs/confirmation" / args.run_id, encode=args.encode))


if __name__ == "__main__":
    main()
