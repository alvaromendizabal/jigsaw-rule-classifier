"""Build the authored-only, two-T4 8B recovery probe without competition data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat

from scripts.build_adapted_notebook import build as original_notebook


def build(root: Path):
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Jigsaw · Qwen3-8B two-GPU recovery probe\n\n"
            "Eight authored comments verify FP16 adapter training, optimizer recovery "
            "and finite inference across two T4 GPUs. This is a software and memory "
            "check only. No competition data, released targets or performance evaluation. "
            "The scored 4B notebook is unchanged.\n\n"
            "Attach the Qwen3-8B mirror `manojkumarcs28/qwen3-8b/pyTorch/qwen3-8b/1`, "
            "select GPU T4 x2 and disable Internet. Every upstream asset hash must pass."
        ),
        original_notebook(root).cells[1],
    ]
    modules = {"jigsaw_rules/__init__.py": "", "scripts/__init__.py": ""}
    for name in ("data.py", "runtime.py"):
        modules["jigsaw_rules/" + name] = (root / "src/jigsaw_rules" / name).read_text()
    for name in (
        "competition_features.py",
        "support_adaptation.py",
        "decision_training.py",
        "kaggle_adaptation.py",
        "verify_kaggle_gpu.py",
    ):
        modules["scripts/" + name] = (root / "scripts" / name).read_text()
    for name in ("LICENSE", "tokenizer_config.json"):
        modules["assets/" + name] = (root / "assets/qwen3_8b" / name).read_text()
    hashes = {}
    for name, content in modules.items():
        hashes[name] = hashlib.sha256(content.encode()).hexdigest()
        cell = nbformat.v4.new_code_cell(f"install_source({name!r}, {content!r}, {hashes[name]!r})")
        cell.metadata["jupyter"] = {"source_hidden": True}
        cells.append(cell)
    model = json.loads((root / "configs/qwen3_8b.json").read_text())
    training = json.loads((root / "configs/backbone_capacity.json").read_text())["training"]
    cells.append(
        nbformat.v4.new_code_cell(
            "import torch, platform\n"
            "from jigsaw_rules.runtime import atomic_json, digest\n"
            "from scripts.competition_features import content_hash\n"
            "from scripts.kaggle_adaptation import prepare_model\n"
            "from scripts.verify_kaggle_gpu import run\n"
            f"model_spec = json.loads({json.dumps(model)!r})\n"
            f"training = json.loads({json.dumps(training)!r})\n"
            f"sources = json.loads({json.dumps(hashes)!r})\n"
            "assert torch.cuda.device_count() == 2, 'Select GPU T4 x2'\n"
            "layout = {'model.embed_tokens': 0, "
            "**{f'model.layers.{i}': 0 if i < 18 else 1 for i in range(36)}, "
            "'model.norm': 1, 'lm_head': 1}\n"
            "contract = {'model': model_spec, 'training': training, 'device_map': layout, "
            "'source': sources, 'python': platform.python_version(), "
            "'packages': {n: version(n) for n in ('torch', 'transformers', 'peft', 'accelerate')}, "
            "'devices': [torch.cuda.get_device_name(i) for i in range(2)]}\n"
            "key = content_hash(contract)[:20]\n"
            "directory = Path('/kaggle/working/capacity_probe') / key\n"
            "marker = directory / 'manifest.json'\n"
            "if marker.exists():\n"
            "    receipt = json.loads(marker.read_text())\n"
            "    assert receipt['contract'] == contract\n"
            "    assert receipt['receipt_sha256'] == digest(directory / 'complete.json')\n"
            "    print('COMPLETED_PROBE_REUSED', key)\n"
            "else:\n"
            "    mirrors = list(Path('/kaggle/input').rglob('model-00001-of-00005.safetensors'))\n"
            "    assert len(mirrors) == 1, 'Attach exactly the registered 8B model mirror'\n"
            "    model = prepare_model(mirrors[0].parent, runtime / 'model', "
            "model_spec, runtime / 'assets')\n"
            "    result = run(model, directory, model_spec, training, device_map=layout)\n"
            "    receipt = {'run_id': key, 'status': 'passed', 'contract': contract, "
            "'result': result, 'receipt_sha256': digest(directory / 'complete.json')}\n"
            "    atomic_json(marker, receipt)\n"
            "atomic_json(Path('/kaggle/working/capacity_probe_manifest.json'), receipt)\n"
            "print(json.dumps(receipt, indent=2))\n"
        )
    )
    return nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    nbformat.write(build(root), root / "kaggle/capacity_probe.ipynb")
