"""Prepare target-free confirmation inputs or replay fixed inference."""

import argparse
from pathlib import Path

from jigsaw_rules.confirmation import prepare_inputs, run_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "predict", "freeze", "score"])
    parser.add_argument("--run-id")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--prediction-commit")
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
    directory = root / "runs/confirmation" / args.run_id
    if args.action == "predict":
        print(run_predictions(root, directory, encode=args.encode))
    elif args.action == "freeze":
        from jigsaw_rules.confirmation_evaluation import prediction_freeze
        from jigsaw_rules.runtime import atomic_json

        output = root / "reports/confirmation/prediction_freeze.json"
        atomic_json(output, prediction_freeze(root, directory))
        print(output)
    else:
        from jigsaw_rules.confirmation_evaluation import evaluate_confirmation

        if not args.prediction_commit:
            raise ValueError("Scoring requires the published prediction-freeze commit")
        print(evaluate_confirmation(root, directory, args.prediction_commit))


if __name__ == "__main__":
    main()
