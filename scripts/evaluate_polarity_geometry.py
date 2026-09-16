from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from jigsaw_rules.polarity_geometry import evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    frame = pd.read_csv(args.train)
    with np.load(args.representations, allow_pickle=False) as data:
        required = {"row_ids", "scores", "vectors"}
        if required.difference(data.files):
            raise ValueError("representation cache missing required arrays")
        result = evaluate(frame, data["row_ids"], data["scores"], data["vectors"])
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "best_candidate": result["best_candidate"],
                "promotion": result["promotion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
