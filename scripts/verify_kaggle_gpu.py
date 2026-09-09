"""Run a small authored-data recovery check on the actual offline Kaggle GPU."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from jigsaw_rules.runtime import Progress, atomic_json
from scripts.competition_features import FrozenEncoder
from scripts.decision_training import train_adapter
from scripts.support_adaptation import add_adapter, decision_prompts


def run(model_path: Path, output: Path, model_spec: dict, training: dict):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="left", local_files_only=True, trust_remote_code=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    rows = [
        {
            "rule": "No Advertising: unsolicited promotional content is not allowed.",
            "body": body,
            "rule_violation": label,
        }
        for body, label in [
            ("Buy my discounted shoes today. Click my shop link for a special offer!", 1),
            ("I liked the discussion of running techniques in this thread.", 0),
            ("Use my referral code to sign up and get a bonus now!", 1),
            ("The article discusses why some running shoes wear out faster.", 0),
            ("Visit my online store this weekend for a limited sale!", 1),
            ("Thank you for sharing your experience with the local running club.", 0),
            ("Subscribe to my paid channel using this promotional discount code.", 1),
            ("Our neighborhood walking trail was muddy after the rain.", 0),
        ]
    ]
    texts = decision_prompts(rows, tokenizer, model_spec)
    spec = {
        **training,
        "precision": "float16",
        "effective_batch": 4,
        "micro_batch": 2,
        "checkpoint_steps": 1,
        "verification": "authored_runtime_probe_only",
    }

    def load():
        torch.manual_seed(spec["seed"])
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        )
        return add_adapter(base, spec)

    with Progress(output / "events.jsonl", "kaggle_gpu_verification") as log:
        model = load()
        args = {
            "tokenizer": tokenizer,
            "texts": texts,
            "labels": [r["rule_violation"] for r in rows],
            "repeats": [1] * len(rows),
            "spec": spec,
            "output": output / "training",
            "log": log,
        }

        def interrupt(path):
            if path.name == "complete.json":
                raise InterruptedError("Intentional GPU optimizer recovery probe")

        interrupted = False
        try:
            train_adapter(model, sync=interrupt, **args)
        except InterruptedError:
            interrupted = True
            log.emit("intentional_restart")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load()
        training_record = train_adapter(model, **args)
        assert training_record["resumed_step"] >= 1
        encoder = FrozenEncoder(model.get_base_model(), tokenizer, device="cuda:0")
        scores, _ = encoder.encode(texts[:2])
        assert np.isfinite(scores).all()
        receipt = {
            "status": "passed",
            "scope": "Authored software probe, not model evaluation",
            "device": torch.cuda.get_device_name(),
            "precision": "float16",
            "intentional_interruption": interrupted,
            "training": training_record,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        atomic_json(output / "complete.json", receipt)
        print(json.dumps(receipt), flush=True)
        del encoder, model
        gc.collect()
        torch.cuda.empty_cache()
    return receipt
