"""Package an explicit target-free inference allowlist, with per-file provenance."""

import argparse
import io
import json
import tarfile
from pathlib import Path

from jigsaw_rules.confirmation import load_protocol, read_inputs
from jigsaw_rules.embeddings import QwenEncoder, content_key, load_spec
from jigsaw_rules.expanded_embeddings import verified_keys
from jigsaw_rules.final_model import verify_stage
from jigsaw_rules.runtime import atomic_json, digest
from jigsaw_rules.semantic import input_texts


def package(root: Path, run_id: str, output: Path) -> dict:
    if len(run_id) != 20 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("Invalid confirmation identifier")
    plan = load_protocol(root)
    directory = root / "runs/confirmation" / run_id
    frame, _ = read_inputs(root, directory)
    spec = load_spec(root)
    contract = QwenEncoder(root, spec).contract
    cache = root / "runs/embeddings" / content_key(contract)[:20]
    keys = verified_keys(cache, contract)
    requested = {content_key(t) for t in input_texts(frame, spec)}
    final = root / "runs/model_validation" / plan["model_run"] / "final"
    if digest(final / "complete.json") != plan["model_stage_sha256"]:
        raise ValueError("Frozen fitted model differs")
    files = {cache / "contract.json"}
    for target in [directory / "inputs", final, *sorted(cache.glob("batch_*"))]:
        verify_stage(target)
        marker = target / "complete.json"
        files.add(marker)
        files.update(target / name for name in json.loads(marker.read_text())["files"])
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError("Unsafe inference input file")
    manifest = {
        "schema": 1,
        "run_id": run_id,
        "purpose": "Target-free protected inference; no solution file or row targets",
        "rows": len(frame),
        "requested_unique_inputs": len(requested),
        "reusable_unique_inputs": len(requested & keys),
        "new_unique_inputs": len(requested - keys),
        "files": {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=path.relative_to(root), recursive=False)
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("confirmation-input-manifest.json")
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    result = {k: v for k, v in manifest.items() if k != "files"}
    result.update(bytes=output.stat().st_size, sha256=digest(output), files=len(files))
    atomic_json(output.with_suffix(".manifest.json"), manifest)
    atomic_json(output.with_suffix(".record.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(package(Path(__file__).resolve().parents[1], args.run_id, args.output)))


if __name__ == "__main__":
    main()
