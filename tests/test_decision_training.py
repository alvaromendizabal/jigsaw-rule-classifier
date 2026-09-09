import json

import pytest

from scripts.decision_training import train_adapter
from tests.test_support_adaptation import SilentLog, TinyTokenizer, tiny_model, training_spec


@pytest.mark.parametrize("precision", ["float32", "float16"])
def test_scaled_optimizer_recovery_is_exact(tmp_path, precision):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict

    torch.set_num_threads(2)
    args = {
        "tokenizer": TinyTokenizer(),
        "texts": [f"authored {i}" for i in range(8)],
        "labels": [0, 1] * 4,
        "repeats": [1] * 8,
        "spec": {**training_spec(), "precision": precision},
        "log": SilentLog(),
    }
    uninterrupted = tiny_model()
    train_adapter(uninterrupted, output=tmp_path / "full", **args)

    def stop(path):
        if path.name == "complete.json":
            raise InterruptedError("Durable optimizer boundary")

    with pytest.raises(InterruptedError):
        train_adapter(tiny_model(), output=tmp_path / "resume", sync=stop, **args)
    resumed = tiny_model()
    record = train_adapter(resumed, output=tmp_path / "resume", **args)
    assert record["resumed_step"] == 1
    assert record["loss_scale"] == (128.0 if precision == "float16" else 1.0)
    expected = get_peft_model_state_dict(uninterrupted, save_embedding_layers=False)
    for key, value in get_peft_model_state_dict(resumed, save_embedding_layers=False).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="contract"):
        train_adapter(
            tiny_model(),
            output=tmp_path / "resume",
            **{
                **args,
                "spec": {**args["spec"], "learning_rate": 0.02},
            },
        )
    marker = tmp_path / "resume/step_000002/complete.json"
    record = json.loads(marker.read_text())
    record["sha256"] = "0" * 64
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="checksum"):
        train_adapter(tiny_model(), output=tmp_path / "resume", **args)


def test_disabled_scaling_preserves_original_training(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import get_peft_model_state_dict

    from scripts.support_adaptation import train_adapter as original

    args = {
        "tokenizer": TinyTokenizer(),
        "texts": [f"authored {i}" for i in range(8)],
        "labels": [0, 1] * 4,
        "repeats": [1] * 8,
        "spec": training_spec(),
        "log": SilentLog(),
    }
    old = tiny_model()
    original(old, output=tmp_path / "old", **args)
    new = tiny_model()
    train_adapter(new, output=tmp_path / "new", **args)
    expected = get_peft_model_state_dict(old, save_embedding_layers=False)
    for key, value in get_peft_model_state_dict(new, save_embedding_layers=False).items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_gradient_overflow_retries_same_step_and_saves_scaler(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    model = tiny_model()
    parameter = next(p for p in model.parameters() if p.requires_grad)
    injected = []

    def overflow_once(gradient):
        if not injected:
            injected.append(True)
            return torch.full_like(gradient, float("inf"))
        return gradient

    parameter.register_hook(overflow_once)
    events = []

    class Recorder:
        def emit(self, event, **fields):
            events.append((event, fields))

    result = train_adapter(
        model,
        TinyTokenizer(),
        ["one", "two", "three", "four"],
        [0, 1, 0, 1],
        [1] * 4,
        {**training_spec(), "precision": "float16"},
        tmp_path,
        Recorder(),
    )
    assert result["optimizer_steps"] == 1 and result["loss_scale"] == 64.0
    assert sum(event == "gradient_overflow_retry" for event, _ in events) == 1
    assert sum(event == "optimizer_step" for event, _ in events) == 1
    state = torch.load(tmp_path / "step_000001/state.pt", weights_only=False)
    assert state["scaler"]["scale"] == 64.0
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_retained_checkpoints_remain_resumable(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    args = {
        "tokenizer": TinyTokenizer(),
        "texts": [f"authored {i}" for i in range(12)],
        "labels": [0, 1] * 6,
        "repeats": [1] * 12,
        "spec": {**training_spec(), "keep_checkpoints": 2},
        "log": SilentLog(),
    }
    train_adapter(tiny_model(), output=tmp_path, **args)
    assert sorted(path.name for path in tmp_path.glob("step_*")) == [
        "step_000002",
        "step_000003",
    ]
    assert train_adapter(tiny_model(), output=tmp_path, **args)["resumed_step"] == 3
