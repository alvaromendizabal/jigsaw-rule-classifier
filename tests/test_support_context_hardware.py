"""Guard the public-only preflight boundary after the private-transfer rejection."""

import ast

from scripts.support_context_hardware import INPUT_SHA256, public_cuda_script


def test_public_probe_has_no_private_storage_or_credentials():
    source = public_cuda_script()
    tree = ast.parse(source)
    compile(tree, "public-hardware-probe", "exec")
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module)
    assert modules == {"json", "platform", "time", "datetime", "torch"}
    for forbidden in ("JIGSAW_PROBE_", INPUT_SHA256, "https://", "s3://", "os.environ"):
        assert forbidden not in source
