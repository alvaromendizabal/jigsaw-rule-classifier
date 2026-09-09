"""Build the offline neural candidate from reviewable canonical source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nbformat


def build(root: Path, destination: Path | None = None):
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Jigsaw · Support-adapted rule classification\n\n"
            "Train one LoRA adapter from the original training comments and the supplied "
            "positive/negative examples, then score the hidden comments offline. The model "
            "is Qwen3-4B-Instruct-2507 at a verified pre-deadline revision.\n\n"
            "Attach the official competition data and the Guanshuo Xu mirror "
            "`wowfattie/qwen3-4b-instruct-2507/transformers/default/1`. Select GPU T4 x2 "
            "and turn Internet off. Save Version → Save & Run All; submit the successful "
            "version's `submission.csv`. The preview contains only ten rows.\n\n"
            "The matched development study improved 0.6146 → 0.7199 AUC on 881 novel "
            "comments. That is not a Kaggle score or a promise of 0.92. The direct model "
            "won the declared feature comparison; rejected coordinate and geometry "
            "readouts are not used here.\n\n"
            "Recovery includes optimizer, scheduler, loss scaler, RNG and data order. "
            "Two completed optimizer checkpoints and verified prediction batches are kept. "
            "Only supplied support labels and original training labels are eligible; "
            "released hidden targets are never loaded."
        )
    ]
    cells.append(
        nbformat.v4.new_code_cell("""from pathlib import Path
import hashlib, json, os, subprocess, sys
from importlib.metadata import PackageNotFoundError, version
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
try:
    incompatible = version('peft') == '0.19.1' and version('torchao') == '0.10.0'
except PackageNotFoundError:
    incompatible = False
if incompatible:
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=True)
    print('Removed an incompatible optional quantization package; this model uses no quantization.')
runtime = Path('/kaggle/working/jigsaw_runtime')
runtime.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(runtime))
def install_source(relative, content, expected):
    path = runtime / relative
    if runtime.resolve() not in path.resolve().parents:
        raise ValueError('Unsafe runtime source path')
    if hashlib.sha256(content.encode()).hexdigest() != expected:
        raise ValueError('Embedded source checksum differs')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
""")
    )
    modules = {
        "jigsaw_rules/__init__.py": "",
        "scripts/__init__.py": "",
        **{
            "jigsaw_rules/" + name: (root / "src/jigsaw_rules" / name).read_text()
            for name in ["data.py", "runtime.py"]
        },
        **{
            "scripts/" + name: (root / "scripts" / name).read_text()
            for name in [
                "competition_features.py",
                "support_adaptation.py",
                "decision_training.py",
                "kaggle_adaptation.py",
                "kaggle_paths.py",
            ]
        },
        **{
            "assets/" + name: (root / "assets/qwen3" / name).read_text()
            for name in ["LICENSE", "tokenizer_config.json"]
        },
    }
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Verified runtime\nThe following cells install the exact repository modules "
            "and upstream license/tokenizer configuration locally. They make no downloads. "
            "The model weights remain in the attached Kaggle input."
        )
    )
    for relative, content in modules.items():
        sha = hashlib.sha256(content.encode()).hexdigest()
        cell = nbformat.v4.new_code_cell(
            f"# Canonical source: {relative}\ninstall_source({relative!r}, {content!r}, {sha!r})"
        )
        cell.metadata["jupyter"] = {"source_hidden": True}
        cells.append(cell)
    full = json.loads((root / "configs/competition_features.json").read_text())
    model_spec = {
        key: full[key] for key in ["model_id", "revision", "files", "field_tokens", "max_tokens"]
    }
    config = json.loads((root / "configs/kaggle_adaptation.json").read_text())
    cells.append(
        nbformat.v4.new_markdown_cell(
            "## Train, score and validate\nThe first run verifies every model file, fits "
            "one fixed epoch and writes `submission.csv`. A rerun reuses intact completed "
            "work. Scores are ranks within each policy, preserving AUC without sigmoid "
            "saturation; they are not calibrated probabilities."
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            "from scripts.kaggle_adaptation import prepare_model, run\n"
            "from scripts.kaggle_paths import submission_input_root\n"
            f"model_spec=json.loads({json.dumps(model_spec)!r})\n"
            f"settings=json.loads({json.dumps(config)!r})\n"
            "mirror=Path('/kaggle/input/models/wowfattie/qwen3-4b-instruct-2507/transformers/default/1')\n"
            "model=prepare_model(mirror,runtime/'model',model_spec,runtime/'assets')\n"
            "submission=run(submission_input_root(Path.cwd()),model,Path('/kaggle/working'),model_spec,settings)\n"
            "print('Validated submission:',submission)\n"
            "print((submission.parent/'submission_manifest.json').read_text())"
        )
    )
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    for cell in cells:
        cell.id = hashlib.sha256((cell.cell_type + cell.source).encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            compile(cell.source, "submission_cell", "exec")
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
    return notebook


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    build(project, project / "runs/competition/kaggle/submission.ipynb")
