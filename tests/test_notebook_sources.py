"""Generated source stays exact while valid executed evidence remains visible."""

import nbformat as nbf

from scripts.build_notebooks import notebook, same_sources, write_notebook


def test_execution_outputs_survive_regeneration(tmp_path):
    expected = notebook([("md", "A result"), ("code", "print(42)")])
    actual = nbf.from_dict(expected)
    actual.cells[1].execution_count = 1
    actual.cells[1].outputs = [nbf.v4.new_output("stream", name="stdout", text="42\n")]
    path = tmp_path / "result.ipynb"
    nbf.write(actual, path)
    before = path.read_bytes()
    write_notebook(path, expected)
    assert path.read_bytes() == before
    assert same_sources(actual, expected)


def test_changed_code_rejects_stale_output(tmp_path):
    original = notebook([("code", "print(42)")])
    original.cells[0].outputs = [nbf.v4.new_output("stream", name="stdout", text="42\n")]
    changed = notebook([("code", "print(43)")])
    path = tmp_path / "result.ipynb"
    nbf.write(original, path)
    assert not same_sources(original, changed)
    write_notebook(path, changed)
    assert nbf.read(path, as_version=4).cells[0].outputs == []


def test_changed_narrative_requires_regeneration():
    first = notebook([("md", "Synthetic"), ("code", "print(42)")])
    second = notebook([("md", "Competition"), ("code", "print(42)")])
    assert not same_sources(first, second)
