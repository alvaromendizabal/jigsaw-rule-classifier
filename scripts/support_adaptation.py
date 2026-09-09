"""Target-free support-adaptation plans and resumable decision-token learning."""

from __future__ import annotations

import io
import json
import random

import numpy as np
import pandas as pd

from jigsaw_rules.data import EXAMPLES, normalize, validate_frame
from jigsaw_rules.runtime import atomic_bytes, atomic_json, digest
from scripts.competition_features import content_hash, prepare_prompts, support_pairs


def build_study(frame: pd.DataFrame) -> dict:
    """Mimic a new rule with legitimate known supports and genuinely novel queries.

    Query eligibility depends on text only. No query label is exported. Every
    query body is excluded from all adaptation sources, including other rules.
    """
    validate_frame(frame, train=True)
    available = frame.drop(columns="rule_violation")
    known_text = {normalize(text) for name in EXAMPLES for text in frame[name]}
    folds = []
    for rule in sorted(frame.rule.unique()):
        eligible = (frame.rule == rule) & ~frame.body.map(normalize).isin(known_text)
        query = frame.loc[eligible].sort_values("row_id")
        if query.empty:
            raise ValueError("No novel-comment evaluation cohort")
        training, audit = support_pairs(
            frame.loc[frame.rule != rule], available, forbidden=query.body
        )
        rows = training[["body", "rule", "rule_violation"]].to_dict("records")
        for row in rows:
            row["repeat"] = 2 if normalize(row["rule"]) == normalize(rule) else 1
        folds.append(
            {
                "rule": rule,
                "training": rows,
                "queries": query[["row_id", "body", "rule"]].to_dict("records"),
                "audit": {
                    **audit,
                    "candidate_query_rows": int((frame.rule == rule).sum()),
                    "novel_query_rows": len(query),
                    "known_support_overlap_excluded": int((frame.rule == rule).sum()) - len(query),
                    "new_rule_training_pairs": sum(row["repeat"] == 2 for row in rows),
                    "epoch_occurrences": sum(row["repeat"] for row in rows),
                },
            }
        )
    plan = {"schema": 1, "protocol": "new_rule_supplied_support_adaptation", "folds": folds}
    validate_study(plan)
    return plan


def validate_study(plan: dict) -> None:
    seen_queries = set()
    for fold in plan["folds"]:
        query_text = set()
        for row in fold["queries"]:
            if set(row) != {"row_id", "body", "rule"}:
                raise ValueError("Query schema must exclude targets")
            if row["row_id"] in seen_queries or row["rule"] != fold["rule"]:
                raise ValueError("Duplicate query ID or inconsistent policy")
            seen_queries.add(row["row_id"])
            query_text.add(normalize(row["body"]))
        keys = set()
        for row in fold["training"]:
            if set(row) != {"body", "rule", "rule_violation", "repeat"}:
                raise ValueError("Unexpected adaptation training schema")
            if row["rule_violation"] not in (0, 1) or row["repeat"] not in (1, 2):
                raise ValueError("Invalid adaptation label or repetition")
            key = (normalize(row["rule"]), normalize(row["body"]))
            if key in keys or key[1] in query_text:
                raise ValueError("Adaptation contains duplicate pairs or evaluation text")
            keys.add(key)


def decision_prompts(rows: list[dict], tokenizer, spec: dict) -> list[str]:
    """Reuse the exact frozen rule-only template for a matched comparison."""
    frame = pd.DataFrame(rows)[["body", "rule"]].assign(
        row_id=np.arange(len(rows)), subreddit="unused", **{name: "unused" for name in EXAMPLES}
    )
    texts, _ = prepare_prompts(frame, tokenizer, spec)
    return texts[::3]


def add_adapter(model, spec: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=spec["rank"],
            lora_alpha=spec["alpha"],
            lora_dropout=spec["dropout"],
            target_modules=spec["target_modules"],
            bias="none",
        ),
    )


def decision_loss(model, inputs, targets):
    """Full-vocabulary loss at the one Yes/No decision position, never template/EOS."""
    import torch

    if not inputs["attention_mask"][:, -1].all():
        raise ValueError("Decision learning requires left padding")
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    return torch.nn.functional.cross_entropy(logits, targets)


def save_training_state(path, model, optimizer, scheduler, step, contract):
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


def restore_training_state(path, model, optimizer, scheduler, contract):
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
        restore_training_state(markers[-1].parent, model, optimizer, scheduler, contract)
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
        trainable_parameters=sum(p.numel() for p in parameters),
    )
    for step in range(start, len(batches)):
        indices = batches[step]
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for offset in range(0, len(indices), spec["micro_batch"]):
            micro = indices[offset : offset + spec["micro_batch"]]
            inputs = tokenizer([texts[i] for i in micro], padding=True, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            targets = torch.tensor([answer_ids[labels[i]][0] for i in micro], device=device)
            loss = decision_loss(model, inputs, targets) * len(micro) / len(indices)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite decision loss")
            loss.backward()
            total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, spec["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
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
            save_training_state(path, model, optimizer, scheduler, step + 1, contract)
            sync(path / "state.pt")
            sync(path / "complete.json")
            log.emit("optimizer_checkpoint", step=step + 1)
    model.eval()
    if spec["gradient_checkpointing"]:
        model.gradient_checkpointing_disable()
    return {"optimizer_steps": len(batches), "resumed_step": start, "training_contract": contract}


def prototype_features(vectors, reference, labels):
    """Two class centroids and nearest known positive/negative comparisons."""
    features, names = [], []
    for label, name in ((0, "permitted"), (1, "violating")):
        selected = reference[np.asarray(labels) == label]
        if not len(selected):
            raise ValueError("Both support classes are required")
        centroid = selected.mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        similarity = vectors @ selected.T
        features.extend(
            [
                vectors @ centroid,
                similarity.max(axis=1),
                np.sort(similarity, axis=1)[:, -min(5, len(selected)) :].mean(axis=1),
            ]
        )
        names.extend([name + "/centroid", name + "/nearest", name + "/top5"])
    matrix = np.column_stack(features)
    return np.column_stack([matrix, matrix[:, 3:] - matrix[:, :3]]), names + [
        "margin/centroid",
        "margin/nearest",
        "margin/top5",
    ]
