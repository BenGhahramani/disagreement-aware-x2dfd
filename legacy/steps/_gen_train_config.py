#!/usr/bin/env python3
"""Generate a training config with ENV overrides.

Reads base YAML (config/train_config.yaml), applies selected ENV vars to
training.* keys, optionally overrides paths.results_dir to LOG_DIR,
writes to an output YAML path.
"""
from __future__ import annotations

import os, sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception as e:
    print(f"[gencfg][ERROR] PyYAML required: {e}", file=sys.stderr)
    sys.exit(2)

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/train_config.yaml")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("logs/steps/tmp/train_config.gen.yaml")

with BASE.open("r", encoding="utf-8") as f:
    cfg: Dict[str, Any] = yaml.safe_load(f) or {}

paths = cfg.get("paths") or {}
log_dir = os.environ.get("LOG_DIR")
if log_dir:
    paths["results_dir"] = log_dir
    cfg["paths"] = paths

tr = cfg.get("training") or {}

def set_if(env: str, key: str, cast):
    v = os.environ.get(env)
    if v is None or v == "":
        return
    try:
        tr[key] = cast(v)
    except Exception:
        pass

set_if("EPOCHS", "num_train_epochs", float)
set_if("TRAIN_BATCH", "per_device_train_batch_size", int)
set_if("EVAL_BATCH", "per_device_eval_batch_size", int)
set_if("GRAD_ACC", "gradient_accumulation_steps", int)
set_if("LR", "learning_rate", float)
set_if("WEIGHT_DECAY", "weight_decay", float)
set_if("WARMUP", "warmup_ratio", float)
set_if("SCHEDULER", "lr_scheduler_type", str)
set_if("LOGGING_STEPS", "logging_steps", int)
set_if("LORA_R", "lora_r", int)
set_if("LORA_ALPHA", "lora_alpha", int)
set_if("MM_PROJECTOR_LR", "mm_projector_lr", float)
set_if("MAX_TOKENS", "model_max_length", int)

vt = os.environ.get("VISION_TOWER")
if vt:
    tr["vision_tower"] = vt

cfg["training"] = tr

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("[gencfg] wrote:", OUT)

