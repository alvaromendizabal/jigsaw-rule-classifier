"""Verify private artifacts, then apply the pre-score feature stopping rule."""

from pathlib import Path

from jigsaw_rules.feature_decision import run_decision

if __package__:
    from .verify_formatting import verify
else:
    from verify_formatting import verify


def main():
    root = Path(__file__).resolve().parents[1]
    print(run_decision(root, verify(root)))


if __name__ == "__main__":
    main()
