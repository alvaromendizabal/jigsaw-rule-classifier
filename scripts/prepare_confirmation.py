"""Audit reserved inputs without reading their targets or generating predictions."""

from pathlib import Path

from jigsaw_rules.confirmation_inputs import prepare_confirmation

if __name__ == "__main__":
    print(prepare_confirmation(Path(__file__).resolve().parents[1]))
