"""Verify the published byte manifest and saved notebook outputs, without training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> dict:
    root = Path(root)
    path = root / "reports/manual_feature_campaign/publication_manifest.json"
    spec = json.loads(path.read_text())
    notebooks = charts = 0
    for name, expected in spec["files"].items():
        parts = PurePosixPath(name)
        if (
            parts.is_absolute()
            or ".." in parts.parts
            or parts.parts[0] not in {"scripts", "tests", "configs", "docs", "notebooks", "reports"}
        ):
            raise ValueError("unsafe public path")
        target = root / name
        if target.is_symlink() or not target.is_file():
            raise ValueError("missing or symlink publication input: " + name)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("published byte mismatch: " + name)
        if name.endswith(".ipynb"):
            book = json.loads(target.read_text())
            code = [c for c in book["cells"] if c["cell_type"] == "code"]
            if not code or any(c.get("execution_count") is None for c in code):
                raise ValueError("unexecuted notebook: " + name)
            if any(o.get("output_type") == "error" for c in code for o in c["outputs"]):
                raise ValueError("notebook contains an error: " + name)
            n = sum(
                "application/vnd.plotly.v1+json" in o.get("data", {})
                for c in code
                for o in c["outputs"]
            )
            if n < 1:
                raise ValueError("expected saved Plotly output")
            charts += n
            notebooks += 1
    if notebooks != spec["notebooks"]:
        raise ValueError("notebook count differs")
    return {"files_verified": len(spec["files"]), "notebooks": notebooks, "plotly": charts}


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), sort_keys=True))
