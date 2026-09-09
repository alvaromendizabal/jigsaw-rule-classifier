"""Run the preregistered semantic formatting study on development data only."""

import argparse
from pathlib import Path

from jigsaw_rules.formatting import run_formatting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode", action="store_true", help="Encode missing inputs only")
    args = parser.parse_args()
    print(run_formatting(Path(__file__).resolve().parents[1], encode=args.encode))


if __name__ == "__main__":
    main()
