"""Run the preregistered nested validation and development-only final fitting."""

from pathlib import Path

from jigsaw_rules.model_validation import run_validation

if __name__ == "__main__":
    print(run_validation(Path(__file__).resolve().parents[1]))
