"""Run one bounded CSLS support-selection audit on recovered cached vectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jigsaw_rules.support_selection_csls import SelectorConfig, promotion_diagnostics, select_fold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=16)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    public = {
        "schema": 1,
        "method": "csls_hubness_corrected_support_selection",
        "query_targets_read": False,
        "folds": [],
    }
    private = {"schema": 1, "query_targets_read": False, "folds": []}

    for i, fold in enumerate(plan["folds"]):
        with np.load(args.representations / f"fold_{i}.npz", allow_pickle=False) as saved:
            train = saved["train_adapted_vectors"]
            query = saved["query_adapted_vectors"]
            ids = saved["query_row_ids"]
        expected = np.asarray([row["row_id"] for row in fold["queries"]])
        if not np.array_equal(ids, expected):
            raise ValueError("query row order differs")
        result = select_fold(fold, train, query, SelectorConfig(k=args.k))
        gate = promotion_diagnostics(result, 0.25)
        public["folds"].append(
            {key: value for key, value in result.items() if key != "selections"} | {"gate": gate}
        )
        private["folds"].append({"fold": i, "selections": result["selections"]})

    (args.output / "audit.json").write_text(json.dumps(public, indent=2) + "\n")
    (args.output / "selected_pairs.private.json").write_text(json.dumps(private, indent=2) + "\n")
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
