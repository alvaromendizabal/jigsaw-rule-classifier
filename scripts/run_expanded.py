"""Run the preregistered development study; optionally extend frozen embeddings."""

import argparse
from pathlib import Path

from jigsaw_rules.expanded import run_expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode", action="store_true", help="Encode missing development inputs")
    options = parser.parse_args()
    print(run_expanded(Path.cwd(), encode=options.encode), flush=True)


if __name__ == "__main__":
    main()
