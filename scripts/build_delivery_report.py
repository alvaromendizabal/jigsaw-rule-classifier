"""Publish and verify the offline bundle's serving and authored-example evidence."""

from __future__ import annotations

import json
from pathlib import Path

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest


def delivery_evidence(root: Path) -> dict:
    folder = root / "reports/delivery"
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["reader_sha256"] != digest(Path(__file__)):
        raise ValueError("Delivery report reader changed")
    for name, expected in metadata["files"].items():
        if digest(folder / name) != expected:
            raise ValueError("Delivery evidence checksum mismatch")
    record = json.loads((folder / "verification.json").read_text())
    bundle = json.loads((folder / "bundle.json").read_text())
    sources = {p.name: digest(p) for p in (root / "src/jigsaw_rules").glob("*.py")}
    if (
        sources != bundle["sources"]
        or record["status"] != "passed"
        or not all(record["checks"].values())
        or record["targets_opened"]
        or record["models_refitted"] != 0
        or record["network_calls"] != 0
        or record["bundle_manifest_sha256"] != digest(folder / "bundle.json")
        or record["source_sha256"] != digest(root / "scripts/verify_offline.py")
        or record["config_sha256"] != digest(root / "configs/delivery.json")
        or record["demo_sha256"] != digest(root / "configs/demo.json")
    ):
        raise ValueError("Offline delivery evidence no longer matches its verified implementation")
    return record


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "reports/delivery"
    record = json.loads((folder / "verification.json").read_text())
    source = root / record["bundle"] / "bundle.json"
    if digest(source) != record["bundle_manifest_sha256"]:
        raise ValueError("Offline bundle manifest changed")
    atomic_bytes(folder / "bundle.json", source.read_bytes())
    atomic_json(
        folder / "metadata.json",
        {
            "schema": 1,
            "reader_sha256": digest(Path(__file__)),
            "files": {name: digest(folder / name) for name in ("verification.json", "bundle.json")},
        },
    )
    delivery_evidence(root)
    print("DELIVERY_EVIDENCE_VERIFIED")


if __name__ == "__main__":
    main()
