"""Prepare the pinned research partition; never score or expose confirmation targets."""

import json
from pathlib import Path

from jigsaw_rules.released import export_release, load_research, prepare_release, released_evidence
from jigsaw_rules.runtime import Progress


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with Progress(root / "logs/released.jsonl", "released_boundary"):
        directory = prepare_release(root)
        export_release(root, directory)
        research = load_research(root)
        evidence = released_evidence(root)
        print(
            json.dumps(
                {
                    "run_id": directory.parent.name,
                    "research_rows": len(research),
                    "boundary": evidence["boundary"],
                    "metric": evidence["metric"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
