#!/usr/bin/env python3
"""Entry point for simulating QA accuracy and managing run artefacts."""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml
import time
import threading
import json

from utils import (
    append_run_log,
    build_question_ranking,
    compute_balanced_accuracy,
    concat_questions,
    evaluate_dataset,
    evaluation_result_to_dict,
    load_image_paths,
    load_questions,
    prepare_run_paths,
    write_json,
    write_text,
)
from utils.progress import ProgressTracker
from utils.evaluation import QAResponse, QuestionStat, EvaluationResult
from utils.model_scoring import resolve_abs_paths
from utils.paths import (
    CONFIG_ROOT_EVAL,
    DATASETS_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    ensure_core_dirs,
)
from utils.lora_inference import lora_infer_conversation_items

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

ensure_core_dirs()

BASE_DIR = PROJECT_ROOT
DEFAULT_CONFIG_PATH = CONFIG_ROOT_EVAL / "config.yaml"
DEFAULT_BASE_MODEL = os.environ.get("X2DFD_BASE_MODEL", "weights/base/llava-v1.5-7b")
DEFAULT_ADAPTER_PATH = "weights/checkpoints/ckpt/FR/llava-v1.5-7b-lora-[small]"


@dataclass
class InferenceOptions:
    model_base: Optional[str]
    adapter_path: Optional[str]
    temperature: float
    top_p: float
    num_beams: int
    max_new_tokens: int

    @property
    def use_lora(self) -> bool:
        return bool(self.adapter_path)

class Config:
    """Configuration class for managing all config parameters."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._load_config()
        self._setup_paths()

    def _load_config(self):
        """Load YAML config file and set defaults."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file does not exist: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.config.setdefault('paths', {})
        self.config.setdefault('files', {})
        self.config.setdefault('run_params', {})
        self.config.setdefault('output', {})
        self.config.setdefault('model', {})

        self.config['paths'].setdefault('data_dir', str(DATASETS_ROOT / 'raw' / 'data'))
        self.config['paths'].setdefault('results_dir', str(OUTPUT_ROOT))
        self.config['paths'].setdefault('image_root_prefix', '')

        self.config['files'].setdefault('question_file', 'question.json')
        self.config['files'].setdefault('real_files', [])
        self.config['files'].setdefault('fake_files', [])

        self.config['run_params'].setdefault('top_k', 3)
        self.config['run_params'].setdefault('use_progress_bar', True)
        self.config['run_params'].setdefault('seed', None)
        self.config['run_params'].setdefault('max_images_per_dataset', None)
        self.config['run_params'].setdefault('truncate_response_tokens', None)

        self.config['output'].setdefault('save_response_records', True)
        self.config['output'].setdefault('save_per_question_balanced', True)
        self.config['output'].setdefault('save_full_ranking', True)

        model_cfg = self.config['model']
        model_cfg.setdefault('base', DEFAULT_BASE_MODEL)
        model_cfg.setdefault('adapter', DEFAULT_ADAPTER_PATH)
        model_cfg.setdefault('temperature', 0.0)
        model_cfg.setdefault('top_p', 1.0)
        model_cfg.setdefault('num_beams', 1)
        model_cfg.setdefault('max_new_tokens', 256)

    def _setup_paths(self):
        self.data_dir = self._coerce_path(self.config['paths']['data_dir'])
        self.results_dir = self._coerce_path(self.config['paths']['results_dir'])
        # Deprecated: image_root_prefix is not used for path resolution anymore
        self.image_root_prefix = self.config['paths']['image_root_prefix']
        self.question_path = self.data_dir / self.config['files']['question_file']
        # For backward compatibility; actual latest manifest is under qa_runs
        self.latest_run_manifest = self.results_dir / "qa_runs" / "latest_run.json"

    def _coerce_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    def _resolve_optional_path_str(self, value: Optional[str]) -> Optional[str]:
        if value is None or value == "":
            return None
        path = Path(value)
        if not path.is_absolute():
            path = BASE_DIR / path
        return str(path)

    @property
    def real_paths(self) -> List[Path]:
        if self.config['files']['real_files']:
            return [self.data_dir / f for f in self.config['files']['real_files']]
        else:
            return [p for p in self.data_dir.glob("real*.json")]

    @property
    def fake_paths(self) -> List[Path]:
        if self.config['files']['fake_files']:
            return [self.data_dir / f for f in self.config['files']['fake_files']]
        else:
            return [p for p in self.data_dir.glob("fake*.json")]

    @property
    def top_k(self) -> int:
        return self.config['run_params']['top_k']

    @property
    def use_progress_bar(self) -> bool:
        return self.config['run_params']['use_progress_bar']

    @property
    def seed(self) -> Optional[int]:
        return self.config['run_params']['seed']

    @property
    def max_images_per_dataset(self) -> Optional[int]:
        return self.config['run_params'].get('max_images_per_dataset')

    @property
    def truncate_response_tokens(self) -> Optional[int]:
        val = self.config['run_params'].get('truncate_response_tokens')
        if isinstance(val, int) and val > 0:
            return val
        return None

    @property
    def model_base_path(self) -> Optional[str]:
        return self._resolve_optional_path_str(self.config['model'].get('base'))

    @property
    def adapter_path(self) -> Optional[str]:
        return self._resolve_optional_path_str(self.config['model'].get('adapter'))

    @property
    def generation_params(self) -> Dict[str, float]:
        model_cfg = self.config['model']
        return {
            'temperature': float(model_cfg.get('temperature', 0.0) or 0.0),
            'top_p': float(model_cfg.get('top_p', 1.0) or 1.0),
            'num_beams': int(model_cfg.get('num_beams', 1) or 1),
            'max_new_tokens': int(model_cfg.get('max_new_tokens', 256) or 256),
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate QA accuracy metrics")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top-k from config file",
    )
    parser.add_argument(
        "--real-files",
        nargs="*",
        default=None,
        help="Override real files from config file",
    )
    parser.add_argument(
        "--fake-files",
        nargs="*",
        default=None,
        help="Override fake files from config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed from config file",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images per dataset (small-batch testing)",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        default=None,
        help="Override base model path for evaluation inference",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter used during evaluation (enables score-based metrics)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override generation temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override nucleus sampling top-p",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=None,
        help="Override beam search width",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override maximum new tokens for generation",
    )
    parser.add_argument(
        "--truncate-response-tokens",
        type=int,
        default=None,
        help="Limit GPT answers to the first N whitespace-separated tokens when recording results.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"Config file does not exist: {args.config}")

    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be greater than 0")

    return args

def _resolve_dataset_image_paths(json_path: Path) -> List[str]:
    """Resolve image paths for a dataset JSON using only top-level Description.

    If Description is missing, all image paths must already be absolute.
    """
    with json_path.open("r", encoding="utf-8") as h:
        data = json.load(h)
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        # Fall back to loader for list of paths (will be treated as-is later)
        from utils.loader import load_image_paths
        rels = load_image_paths(json_path)
        # No Description available; ensure they are absolute or keep as-is (no prefix)
        if any(p and not Path(p).is_absolute() for p in rels):
            # Keep behavior minimal: do not attempt to prefix here
            return [str(Path(p)) for p in rels]
        return [str(Path(p)) for p in rels]
    desc = data.get("Description") or data.get("description")
    desc = desc.strip() if isinstance(desc, str) else None
    rels: List[str] = []
    for it in data["images"]:
        if isinstance(it, dict):
            p = it.get("image_path") or it.get("path")
            if isinstance(p, str):
                rels.append(p)
    if desc:
        return resolve_abs_paths(rels, root_prefix=desc)
    # No Description: require absolute or pass through
    return resolve_abs_paths(rels, root_prefix=None)

def build_dataset_metrics(real_results, fake_results) -> Dict[str, Dict[str, object]]:
    return {
        "real": [evaluation_result_to_dict(r) for r in real_results],
        "fake": [evaluation_result_to_dict(r) for r in fake_results],
    }

def build_response_records(dataset_label: str, responses, use_progress_bar: bool = True) -> Dict[str, object]:
    records = []
    iterable = responses
    if tqdm is not None and use_progress_bar:
        print(f"Generating response records for {dataset_label} dataset ({len(responses)} entries)...")
        iterable = tqdm(responses, desc=f"Processing {dataset_label} responses", ncols=80)
    for entry in iterable:
        # Use the path as produced by inference (should already be absolute under new rules)
        image_path = entry.image_path
        rec = {
            "image_path": image_path,
            "question": entry.question,
            "answer": entry.answer,
        }
        if entry.score is not None:
            rec["score"] = entry.score
        records.append(rec)
    return {"dataset": dataset_label, "responses": records}

def average(lst: List[float]) -> float:
    if not lst:
        return 0.0
    return sum(lst) / len(lst)

def merge_question_stats(stats_list: List[List]) -> List:
    """
    Merge question_stats from multiple datasets by averaging their metrics.
    Assumes all stats_list have the same questions in the same order.
    Returns QuestionStat objects for compatibility with build_question_ranking.
    """
    if not stats_list:
        return []
    
    # Import QuestionStat here to avoid circular imports
    from utils.evaluation import QuestionStat
    
    merged: List[QuestionStat] = []
    num_questions = len(stats_list[0])
    for i in range(num_questions):
        first_stat = stats_list[0][i]

        # Sum counts across datasets to avoid rounding distortion
        if hasattr(first_stat, 'question'):
            question_text = first_stat.question  # type: ignore[attr-defined]
            yes_sum = sum(stats[i].yes_count for stats in stats_list)
            no_sum = sum(stats[i].no_count for stats in stats_list)
            total_sum = sum(stats[i].total_count for stats in stats_list)
        else:
            question_text = first_stat["question"]
            yes_sum = sum(stats[i]["yes_count"] for stats in stats_list)
            no_sum = sum(stats[i]["no_count"] for stats in stats_list)
            total_sum = sum(stats[i]["total_count"] for stats in stats_list)

        merged.append(
            QuestionStat(
                question=question_text,
                yes_count=yes_sum,
                no_count=no_sum,
                total_count=total_sum,
                accuracy=0.0,  # not used directly; ranking recomputes accuracies
            )
        )
    return merged

def _truncate_tokens(text: str, limit: Optional[int]) -> str:
    if not text or not limit:
        return text
    tokens = text.split()
    if len(tokens) <= limit:
        return text
    return " ".join(tokens[:limit])


def _run_inference(items: List[Dict[str, Any]], inference: InferenceOptions) -> List[Dict[str, Any]]:
    if not inference.use_lora:
        raise SystemExit("LoRA adapter is required for evaluation; please train first or set model.adapter in config.")
    if not inference.adapter_path:
        raise SystemExit("adapter_path missing while attempting LoRA inference")
    if not inference.model_base:
        raise SystemExit("model_base is required for LoRA inference")
    return lora_infer_conversation_items(
        items,
        image_root_prefix=None,
        model_path=inference.adapter_path,
        model_base=inference.model_base,
        temperature=inference.temperature,
        top_p=inference.top_p,
        num_beams=inference.num_beams,
        max_new_tokens=inference.max_new_tokens,
        add_scores_turns=True,
    )


def _extract_score(conversations: List[Dict[str, Any]]) -> Optional[float]:
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).lower() == "fake score":
            val = turn.get("value")
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _collect_score_pairs(responses: List[QAResponse], label: int) -> List[Tuple[float, int]]:
    pairs: List[Tuple[float, int]] = []
    for resp in responses:
        if resp.score is not None:
            pairs.append((resp.score, label))
    return pairs


def compute_auc(score_label_pairs: List[Tuple[float, int]]) -> Optional[float]:
    if not score_label_pairs:
        return None
    pos = sum(1 for _, lbl in score_label_pairs if lbl == 1)
    neg = sum(1 for _, lbl in score_label_pairs if lbl == 0)
    if pos == 0 or neg == 0:
        return None
    sorted_pairs = sorted(score_label_pairs, key=lambda x: x[0])
    rank = 1
    rank_sum_pos = 0.0
    n = len(sorted_pairs)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        group_size = j - i
        avg_rank = (2 * rank + group_size - 1) / 2.0
        pos_in_group = sum(1 for _, lbl in sorted_pairs[i:j] if lbl == 1)
        rank_sum_pos += avg_rank * pos_in_group
        rank += group_size
        i = j
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return auc

def load_conversation_pairs(path: Path) -> List[tuple[str, str]]:
    """Load (image, question) pairs from a conversation-style JSON list."""
    import json
    with path.open("r", encoding="utf-8") as h:
        data = json.load(h)
    if not isinstance(data, list):
        return []
    pairs: List[tuple[str, str]] = []
    for item in data:
        try:
            image_rel = item.get("image")
            conv = item.get("conversations") or []
            question = None
            for turn in conv:
                if isinstance(turn, dict) and turn.get("from") == "human":
                    val = turn.get("value")
                    if isinstance(val, str) and val.strip():
                        question = val
                        break
            if isinstance(image_rel, str) and question:
                pairs.append((image_rel, question))
        except Exception:
            continue
    return pairs

def evaluate_pairs(
    pairs: List[tuple[str, str]],
    image_root_prefix: Optional[str],
    *,
    inference: InferenceOptions,
    truncate_tokens: Optional[int] = None,
) -> EvaluationResult:
    """Batch-evaluate explicit (image, question) pairs via conversation JSON."""
    # Build conversation items template
    items = []
    for idx, (image_path, question) in enumerate(pairs, start=1):
        items.append({
            "id": str(idx),
            "image": image_path,
            "conversations": [
                {"from": "human", "value": question},
                {"from": "gpt", "value": ""},
            ],
        })

    answered = _run_inference(items, inference)

    from collections import defaultdict
    yes_count = 0
    no_count = 0
    total_count = 0
    yes_by_q = defaultdict(int)
    no_by_q = defaultdict(int)
    total_by_q = defaultdict(int)
    responses: List[QAResponse] = []

    for it in answered:
        image = it.get("image")
        conv = it.get("conversations") or []
        q = ""
        a = ""
        for turn in conv:
            if isinstance(turn, dict) and turn.get("from") == "human":
                q = turn.get("value") or q
            if isinstance(turn, dict) and turn.get("from") == "gpt":
                a = turn.get("value") or a
        score = _extract_score(conv)
        a_trunc = _truncate_tokens(a, truncate_tokens)
        responses.append(QAResponse(image_path=image, question=q, answer=a_trunc, score=score))
        ans_norm = (a_trunc or "").strip().lower().rstrip(".!?,")
        if ans_norm == "yes":
            yes_count += 1
            yes_by_q[q] += 1
        elif ans_norm == "no":
            no_count += 1
            no_by_q[q] += 1
        total_count += 1
        total_by_q[q] += 1

    stats: List[QuestionStat] = []
    for q in total_by_q.keys():
        stats.append(QuestionStat(question=q, yes_count=yes_by_q[q], no_count=no_by_q[q], total_count=total_by_q[q], accuracy=0.0))

    acc = yes_count / total_count if total_count else 0.0
    return EvaluationResult(yes_count=yes_count, no_count=no_count, total_count=total_count, accuracy=acc, image_count=0, question_count=0, question_stats=stats, responses=responses)

    # ProgressTracker moved to utils.progress.ProgressTracker

def main() -> None:
    args = parse_args()

    config = Config(args.config)

    if args.top_k is not None:
        config.config['run_params']['top_k'] = args.top_k
    if args.real_files is not None:
        config.config['files']['real_files'] = args.real_files
    if args.fake_files is not None:
        config.config['files']['fake_files'] = args.fake_files
    if args.seed is not None:
        config.config['run_params']['seed'] = args.seed
    if args.max_images is not None:
        config.config['run_params']['max_images_per_dataset'] = args.max_images
    if args.truncate_response_tokens is not None:
        config.config['run_params']['truncate_response_tokens'] = args.truncate_response_tokens
    if args.model_base is not None:
        config.config['model']['base'] = args.model_base
    if args.adapter_path is not None:
        config.config['model']['adapter'] = args.adapter_path
    if args.temperature is not None:
        config.config['model']['temperature'] = args.temperature
    if args.top_p is not None:
        config.config['model']['top_p'] = args.top_p
    if args.num_beams is not None:
        config.config['model']['num_beams'] = args.num_beams
    if args.max_new_tokens is not None:
        config.config['model']['max_new_tokens'] = args.max_new_tokens

    truncate_tokens = config.truncate_response_tokens

    inference_opts = InferenceOptions(
        model_base=config.model_base_path,
        adapter_path=config.adapter_path,
        temperature=config.generation_params['temperature'],
        top_p=config.generation_params['top_p'],
        num_beams=int(config.generation_params['num_beams']),
        max_new_tokens=int(config.generation_params['max_new_tokens']),
    )

    if not inference_opts.use_lora:
        print("[WARN] No LoRA adapter configured; evaluation will use base model and skip AUC computation.")

    print("Loading questions and image paths...")
    questions = load_questions(config.question_path)

    real_paths = config.real_paths
    fake_paths = config.fake_paths

    if not real_paths:
        raise RuntimeError("No real dataset files found.")
    if not fake_paths:
        raise RuntimeError("No fake dataset files found.")

    # Support two input schemas:
    # 1) Dataset JSON: {"images": [{"image_path": "..."}]}
    # 2) Conversation JSON: [{"id":..., "image":..., "conversations": [...]}]
    real_datasets = []
    fake_datasets = []
    for p in real_paths:
        try:
            pairs = load_conversation_pairs(p)
            if pairs:
                real_datasets.append({"type": "pairs", "path": p, "pairs": pairs})
            else:
                imgs = _resolve_dataset_image_paths(p)
                real_datasets.append({"type": "images", "path": p, "images": imgs})
        except Exception:
            imgs = _resolve_dataset_image_paths(p)
            real_datasets.append({"type": "images", "path": p, "images": imgs})
    for p in fake_paths:
        try:
            pairs = load_conversation_pairs(p)
            if pairs:
                fake_datasets.append({"type": "pairs", "path": p, "pairs": pairs})
            else:
                imgs = _resolve_dataset_image_paths(p)
                fake_datasets.append({"type": "images", "path": p, "images": imgs})
        except Exception:
            imgs = _resolve_dataset_image_paths(p)
            fake_datasets.append({"type": "images", "path": p, "images": imgs})

    # Apply small-batch limit if configured
    if config.max_images_per_dataset is not None and config.max_images_per_dataset > 0:
        limit = config.max_images_per_dataset
        for ds in real_datasets:
            if ds["type"] == "images":
                ds["images"] = ds["images"][:limit]
            else:
                ds["pairs"] = ds["pairs"][:limit]
        for ds in fake_datasets:
            if ds["type"] == "images":
                ds["images"] = ds["images"][:limit]
            else:
                ds["pairs"] = ds["pairs"][:limit]

    if config.seed is not None:
        seed = config.seed
    else:
        system_rng = random.SystemRandom()
        seed = system_rng.randrange(2**32)
    # RNG previously used for narrative generation; no longer needed

    # Progress file path (organized under qa_runs)
    progress_path = config.results_dir / "qa_runs" / "progress.json"
    progress_tracker = ProgressTracker(progress_path)
    progress_tracker.start()

    # Prepare run directory early so per-dataset artefacts can be saved immediately
    top_k = config.top_k
    run_paths = prepare_run_paths(config.results_dir, top_k)
        # AUC computation removed: no longer collect score-label pairs

    try:
        print("Evaluating fake datasets...")
        fake_results = []
        for idx, ds in enumerate(fake_datasets):
            label = f"Fake dataset [{idx+1}/{len(fake_datasets)}]"
            progress_key = f"fake_{idx+1}"
            if ds["type"] == "images":
                images = ds["images"]  # absolute paths
                pairs = [(img, f"<image>\n{q}") for img in images for q in questions]
                total = len(pairs)
                progress_tracker.update(progress_key, {"current": 0, "total": total, "desc": label})
                result = evaluate_pairs(pairs, image_root_prefix=None, inference=inference_opts, truncate_tokens=truncate_tokens)
            else:
                pairs_raw = ds["pairs"]  # conversation-style JSON should already be absolute under new rules
                pairs = [(p, q) for p, q in pairs_raw]
                total = len(pairs)
                progress_tracker.update(progress_key, {"current": 0, "total": total, "desc": label})
                result = evaluate_pairs(pairs, image_root_prefix=None, inference=inference_opts, truncate_tokens=truncate_tokens)
            fake_results.append(result)
            # AUC computation removed
            
            # Save per-dataset responses and stats immediately after finishing this dataset
            if config.config['output']['save_response_records']:
                dataset_name = fake_paths[idx].stem
                resp = build_response_records(f"fake_{idx+1}", result.responses, config.use_progress_bar)
                per_dataset_resp = resp["responses"]
                write_json(run_paths.responses_fake_dir / f"{dataset_name}_responses.json", {"dataset": dataset_name, "responses": per_dataset_resp})
                stats_payload = [
                    {
                        "question": s.question,
                        "yes_count": s.yes_count,
                        "no_count": s.no_count,
                        "total_count": s.total_count,
                    }
                    for s in result.question_stats
                ]
                write_json(run_paths.stats_fake_dir / f"{dataset_name}_question_stats.json", {"dataset": dataset_name, "question_stats": stats_payload})
            progress_tracker.update(progress_key, {"current": total, "total": total, "desc": label})

        print("Evaluating real datasets...")
        real_results = []
        for idx, ds in enumerate(real_datasets):
            label = f"Real dataset [{idx+1}/{len(real_datasets)}]"
            progress_key = f"real_{idx+1}"
            if ds["type"] == "images":
                images = ds["images"]
                pairs = [(img, f"<image>\n{q}") for img in images for q in questions]
                total = len(pairs)
                progress_tracker.update(progress_key, {"current": 0, "total": total, "desc": label})
                result = evaluate_pairs(pairs, image_root_prefix=None, inference=inference_opts, truncate_tokens=truncate_tokens)
            else:
                pairs_raw = ds["pairs"]
                # Conversation-style JSON must already carry absolute image paths under new rules
                pairs = [(p, q) for p, q in pairs_raw]
                total = len(pairs)
                progress_tracker.update(progress_key, {"current": 0, "total": total, "desc": label})
                result = evaluate_pairs(pairs, image_root_prefix=None, inference=inference_opts, truncate_tokens=truncate_tokens)
            real_results.append(result)
            # AUC computation removed

            # Save per-dataset responses and stats immediately after finishing this dataset
            if config.config['output']['save_response_records']:
                dataset_name = real_paths[idx].stem
                resp = build_response_records(f"real_{idx+1}", result.responses, config.use_progress_bar)
                per_dataset_resp = resp["responses"]
                write_json(run_paths.responses_real_dir / f"{dataset_name}_responses.json", {"dataset": dataset_name, "responses": per_dataset_resp})
                stats_payload = [
                    {
                        "question": s.question,
                        "yes_count": s.yes_count,
                        "no_count": s.no_count,
                        "total_count": s.total_count,
                    }
                    for s in result.question_stats
                ]
                write_json(run_paths.stats_real_dir / f"{dataset_name}_question_stats.json", {"dataset": dataset_name, "question_stats": stats_payload})
            progress_tracker.update(progress_key, {"current": total, "total": total, "desc": label})

        # Compute weighted accuracies across datasets
        total_real = sum(r.total_count for r in real_results)
        total_fake = sum(r.total_count for r in fake_results)
        real_correct = sum(r.no_count for r in real_results)  # real => expect "no"
        fake_correct = sum(r.yes_count for r in fake_results)  # fake => expect "yes"
        real_acc = (real_correct / total_real) if total_real else 0.0
        fake_acc = (fake_correct / total_fake) if total_fake else 0.0
        balanced_acc = compute_balanced_accuracy(real_acc, fake_acc)
        # top_k already set earlier when preparing run paths

        merged_real_stats = merge_question_stats([r.question_stats for r in real_results])
        merged_fake_stats = merge_question_stats([r.question_stats for r in fake_results])
        full_ranking = build_question_ranking(merged_real_stats, merged_fake_stats, top_k=None)
        ranking = full_ranking[:top_k]
        ranking_questions = [entry["question"] for entry in ranking]
        concatenated_questions = concat_questions(ranking_questions)
        per_question_balanced = {entry["question"]: entry["balanced_accuracy"] for entry in full_ranking}

        if config.config['output']['save_response_records']:
            print("Generating response records for real datasets...")
            real_responses_all = []
            for idx, r in enumerate(real_results):
                resp = build_response_records(f"real_{idx+1}", r.responses, config.use_progress_bar)
                real_responses_all.append(resp)
            print("Generating response records for fake datasets...")
            fake_responses_all = []
            for idx, r in enumerate(fake_results):
                resp = build_response_records(f"fake_{idx+1}", r.responses, config.use_progress_bar)
                fake_responses_all.append(resp)
        else:
            real_responses_all = []
            fake_responses_all = []

        print("Saving evaluation results...")
        dataset_metrics = build_dataset_metrics(real_results, fake_results)
        if config.config['output']['save_response_records']:
            for idx, resp in enumerate(real_responses_all):
                dataset_metrics["real"][idx]["responses"] = resp["responses"]
            for idx, resp in enumerate(fake_responses_all):
                dataset_metrics["fake"][idx]["responses"] = resp["responses"]
        write_json(run_paths.dataset_metrics_path, dataset_metrics)

        if config.config['output']['save_per_question_balanced']:
            write_json(run_paths.per_question_balanced_path, per_question_balanced)

        if config.config['output']['save_full_ranking']:
            write_json(run_paths.question_ranking_all_path, {"questions": full_ranking})

        write_json(run_paths.question_ranking_path, {"questions": ranking})
        write_text(run_paths.concatenated_questions_path, concatenated_questions)

        # Per-dataset responses and stats were already saved right after each dataset finished

        summary_payload = {
            "run_id": run_paths.run_id,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "seed": seed,
            "top_k": top_k,
            "balanced_accuracy": balanced_acc,
            "datasets": {
                "real_accuracy": real_acc,
                "fake_accuracy": fake_acc,
            },
            "artefacts": {
                "dataset_metrics": str(run_paths.dataset_metrics_path),
                "per_question_balanced": str(run_paths.per_question_balanced_path),
                "question_ranking_all": str(run_paths.question_ranking_all_path),
                "question_ranking_top_k": str(run_paths.question_ranking_path),
                "concatenated_questions": str(run_paths.concatenated_questions_path),
            },
        }
        write_json(run_paths.summary_path, summary_payload)

        write_json(
            run_paths.settings_path,
            {
                "config_file": str(config.config_path),
                "question_file": str(config.question_path),
                "real_files": [str(p) for p in real_paths],
                "fake_files": [str(p) for p in fake_paths],
                "seed": seed,
                "top_k": top_k,
                "run_id": run_paths.run_id,
                # Path prefix deprecated; paths are resolved via JSON Description or absolute inputs
            },
        )

        append_run_log(
            run_paths.run_log_path,
            timestamp=datetime.utcnow(),
            seed=seed,
            top_k=top_k,
            real_acc=real_acc,
            fake_acc=fake_acc,
            balanced_acc=balanced_acc,
            run_id=run_paths.run_id,
        )

        write_json(
            run_paths.latest_manifest_path,
            {
                "run_id": run_paths.run_id,
                "timestamp": summary_payload["timestamp"],
                "top_k": top_k,
                "balanced_accuracy": balanced_acc,
                "summary": str(run_paths.summary_path),
                "dataset_metrics": str(run_paths.dataset_metrics_path),
                "question_ranking_all": str(run_paths.question_ranking_all_path),
                "question_ranking_top_k": str(run_paths.question_ranking_path),
                "per_question_balanced": str(run_paths.per_question_balanced_path),
                "concatenated_questions": str(run_paths.concatenated_questions_path),
                "responses_real_dir": str(run_paths.responses_real_dir),
                "responses_fake_dir": str(run_paths.responses_fake_dir),
                "stats_real_dir": str(run_paths.stats_real_dir),
                "stats_fake_dir": str(run_paths.stats_fake_dir),
            },
        )

        print(f"Run ID: {run_paths.run_id}")
        print(f"Fake ACC: {fake_acc:.4f} (Averaged over {len(fake_results)} fake datasets)")
        print(f"Real ACC: {real_acc:.4f} (Averaged over {len(real_results)} real datasets)")
        print(f"Balanced ACC: {balanced_acc:.4f}")
        print(f"Top-{top_k} questions by balanced accuracy: {ranking_questions}")
        if config.config['output']['save_response_records']:
            print(f"Per-dataset responses saved under: {run_paths.responses_dir}")
        print(f"Summary saved to: {run_paths.summary_path}")
        print(f"Run log updated: {run_paths.run_log_path}")
        print(f"Config file used: {config.config_path}")
    finally:
        progress_tracker.stop()

if __name__ == "__main__":
    main()
