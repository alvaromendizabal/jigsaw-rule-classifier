"""Mixed-precision decision-token training with complete optimizer recovery.

The original BF16 study remains immutable in support_adaptation.py. This runtime
adds FP16 loss scaling for Kaggle T4 hardware and records that precision contract.
"""

from __future__ import annotations

import io
import json
import random
import shutil

import numpy as np

from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest
from scripts.competition_features import content_hash


def decision_loss(model, inputs, targets):
    """Full-vocabulary loss at the one Yes/No decision position, never template/EOS."""
    import torch

    if not inputs["attention_mask"][:, -1].all():
        raise ValueError("Decision learning requires left padding")
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    return torch.nn.functional.cross_entropy(logits, targets)


def save_training_state(path, model, optimizer, scheduler, step, contract, scaler):
    import torch
    from peft import get_peft_model_state_dict

    state = {
        "contract": contract,
        "step": step,
        "adapter": {
            k: v.cpu()
            for k, v in get_peft_model_state_dict(model, save_embedding_layers=False).items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    payload = io.BytesIO()
    torch.save(state, payload)
    atomic_bytes(path / "state.pt", payload.getvalue())
    atomic_json(
        path / "complete.json",
        {"step": step, "contract": contract, "sha256": digest(path / "state.pt")},
    )


def restore_training_state(path, model, optimizer, scheduler, contract, scaler):
    import torch
    from peft import set_peft_model_state_dict

    marker = json.loads((path / "complete.json").read_text())
    if marker["contract"] != contract or digest(path / "state.pt") != marker["sha256"]:
        raise ValueError("Training checkpoint contract/checksum differs")
    # Only our own checksum-verified private training checkpoint is deserialized.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if state["contract"] != contract or state["step"] != marker["step"]:
        raise ValueError("Training checkpoint payload differs from marker")
    set_peft_model_state_dict(model, state["adapter"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


def train_adapter(model, tokenizer, texts, labels, repeats, spec, output, log, sync=lambda p: None):
    """One fixed epoch; checkpoint only at complete optimizer-step boundaries."""
    import torch

    order = np.repeat(np.arange(len(texts)), repeats)
    np.random.default_rng(spec["seed"]).shuffle(order)
    batches = [
        order[i : i + spec["effective_batch"]]
        for i in range(0, len(order), spec["effective_batch"])
    ]
    contract = content_hash(
        {"texts": texts, "labels": labels, "order": order.tolist(), "spec": spec}
    )
    precision = spec.get("precision", "float32")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Unsupported training precision")
    keep = spec.get("keep_checkpoints")
    if keep is not None and (not isinstance(keep, int) or isinstance(keep, bool) or keep < 1):
        raise ValueError("Checkpoint retention must be a positive integer")
    device = next(model.parameters()).device
    scaler = torch.amp.GradScaler(device.type, enabled=precision == "float16", init_scale=128.0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
    )
    warmup = max(1, int(len(batches) * spec["warmup_fraction"]))

    def schedule(step):
        return min(
            (step + 1) / warmup, max(0.0, (len(batches) - step) / max(1, len(batches) - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    markers = sorted(output.glob("step_*/complete.json"))
    start = (
        restore_training_state(markers[-1].parent, model, optimizer, scheduler, contract, scaler)
        if markers
        else 0
    )
    if start > len(batches):
        raise ValueError("Checkpoint lies beyond the declared training epoch")
    if spec["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    answer_ids = [tokenizer.encode(word, add_special_tokens=False) for word in ("No", "Yes")]
    if any(len(ids) != 1 for ids in answer_ids):
        raise ValueError("Training decisions must be single tokens")
    model.train()
    log.emit(
        "training_ready",
        examples=len(order),
        optimizer_steps=len(batches),
        resumed_step=start,
        precision=precision,
        loss_scale=scaler.get_scale(),
        trainable_parameters=sum(p.numel() for p in parameters),
    )
    for step in range(start, len(batches)):
        indices = batches[step]
        before_python, before_numpy = random.getstate(), np.random.get_state()
        before_rng = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        for attempt in range(9):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for offset in range(0, len(indices), spec["micro_batch"]):
                micro = indices[offset : offset + spec["micro_batch"]]
                inputs = tokenizer([texts[i] for i in micro], padding=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                targets = torch.tensor([answer_ids[labels[i]][0] for i in micro], device=device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
                ):
                    loss = decision_loss(model, inputs, targets) * len(micro) / len(indices)
                if not torch.isfinite(loss):
                    raise ValueError(
                        "Nonfinite forward loss; loss scaling cannot recover activations"
                    )
                scaler.scale(loss).backward()
                total_loss += float(loss.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, spec["gradient_clip"], error_if_nonfinite=not scaler.is_enabled()
            )
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                break
            log.emit(
                "gradient_overflow_retry",
                step=step + 1,
                attempt=attempt + 1,
                scale=scaler.get_scale(),
            )
            random.setstate(before_python)
            np.random.set_state(before_numpy)
            torch.set_rng_state(before_rng)
            if before_cuda:
                torch.cuda.set_rng_state_all(before_cuda)
        else:
            raise ValueError("Repeated FP16 gradient overflow; training stopped before advancing")
        scheduler.step()
        log.emit(
            "optimizer_step",
            step=step + 1,
            total=len(batches),
            loss=total_loss,
            gradient_norm=float(norm),
        )
        if (step + 1) % spec["checkpoint_steps"] == 0 or step + 1 == len(batches):
            path = output / f"step_{step + 1:06d}"
            save_training_state(path, model, optimizer, scheduler, step + 1, contract, scaler)
            sync(path / "state.pt")
            sync(path / "complete.json")
            log.emit("optimizer_checkpoint", step=step + 1)
            if keep is not None:
                for old in sorted(output.glob("step_*/complete.json"))[:-keep]:
                    if old.parent.parent != output or old.parent.is_symlink():
                        raise ValueError("Unsafe checkpoint retention path")
                    shutil.rmtree(old.parent)
    model.eval()
    if spec["gradient_checkpointing"]:
        model.gradient_checkpointing_disable()
    return {
        "optimizer_steps": len(batches),
        "resumed_step": start,
        "training_contract": contract,
        "loss_scale": scaler.get_scale(),
    }
