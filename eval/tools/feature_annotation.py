#!/usr/bin/env python3
"""
Generate prompt-based explanations conditioned on real/fake labels.

Notes:
- The path prefix is taken only from the input JSON top-level Description;
  config.paths.image_root_prefix is not used.
- Parameters are provided by config.yaml; only --config is accepted in CLI.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional
import argparse
from datetime import datetime
import yaml
from utils.inference import infer_conversation_items
from utils.progress import ProgressTracker
from utils.model_scoring import get_provider
from utils.paths import OUTPUT_ROOT, PROJECT_ROOT, ensure_core_dirs
from utils.annotation_utils import (
    build_binary_question,
    compose_labeled_response,
)

def load_prompt_template(prompt_file: str) -> str:
    """Load a prompt template JSON with a 'prompt' field."""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    return prompt_data['prompt']

## NOTE: generate_explanation was not used in this script and has been removed.

def process_json_file(
    input_file: str,
    output_file: str,
    prompt_template: str,
    model_path: Optional[str] = None,
    max_items: Optional[int] = None,
    label_override: Optional[str] = None,
    image_root_prefix: Optional[str] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    progress_key: Optional[str] = None,
    template_output_file: Optional[str] = None,
):
    """Build a conversation template, run batch inference, and write outputs.

    Accepts two input schemas:
    - Conversation list: [{'id','image','conversations': [...]}]
    - Dataset dict: {"Description": "/abs/root", "images": [{"image_path": "rel/or/abs"}, ...]}

    Behavior:
    - For dataset schema, only use JSON Description as the root prefix.
    - For conversation schema, 'image' must already be absolute; otherwise raise.
    - Normalize all 'image' paths to absolute in the template and final outputs.
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build template items
    template_items = []
    # Helper: strict path absolutization
    # - Dataset schema: require JSON Description when relative paths exist
    # - Conversation schema: require absolute paths
    def _abs_path_dataset(p: Optional[str], root_hint: Optional[str]) -> Optional[str]:
        if not p:
            return None
        if os.path.isabs(p):
            return os.path.normpath(p)
        if root_hint:
            return os.path.normpath(os.path.join(root_hint, p))
        raise ValueError("Relative image_path encountered without JSON Description root")
    if isinstance(data, list):
        items = data[: max_items or len(data)]
        for it in items:
            conv = (it.get('conversations') or [])
            has_gpt = any(isinstance(t, dict) and t.get('from') == 'gpt' for t in conv)
            if not has_gpt:
                conv = list(conv) + [{"from": "gpt", "value": ""}]
            # Conversation schema requires absolute image path
            img_val = it.get('image')
            if not isinstance(img_val, str) or not img_val:
                raise ValueError("Conversation item missing 'image' path")
            if not os.path.isabs(img_val):
                raise ValueError("Conversation-style JSON requires absolute 'image' paths or use Dataset style with Description")
            img_abs = os.path.normpath(img_val)
            template_items.append({
                "id": str(it.get('id', '')),
                "image": img_abs,
                "conversations": conv,
            })
    elif isinstance(data, dict) and 'images' in data:
        # Only use JSON-level Description; no fallback to config prefix
        local_root = None
        try:
            desc = data.get('Description') or data.get('description')
            if isinstance(desc, str) and desc.strip():
                local_root = desc.strip()
        except Exception:
            local_root = None
        # Do not allow relative paths without Description (error below if seen)

        items = data['images'][: max_items or len(data['images'])]
        for idx, it in enumerate(items, start=1):
            image_path = it.get('image_path')
            if not image_path:
                continue
            label = label_override or ('real' if 'real' in os.path.basename(input_file).lower() else 'fake')
            personalized_prompt = prompt_template.replace('{label}', label)
            full_prompt = f"<image>\n{personalized_prompt}"
            # Write absolute image path to the result
            img_abs = _abs_path_dataset(image_path, local_root)
            template_items.append({
                "id": str(idx),
                "image": img_abs,
                "conversations": [
                    {"from": "human", "value": full_prompt},
                    {"from": "gpt", "value": ""},
                ],
            })
    else:
        print(f"Unknown JSON format: {type(data)}")
        return []

    if template_output_file:
        with open(template_output_file, 'w', encoding='utf-8') as f:
            json.dump(template_items, f, ensure_ascii=False, indent=2)

    total = len(template_items)
    updated = infer_conversation_items(
        template_items,
        image_root_prefix=None,  # Force absolute paths; no prefix concatenation
        model_path=model_path or "",
        max_new_tokens=512,
    )
    for i in range(total):
        if progress_tracker and progress_key:
            progress_tracker.update(progress_key, {"current": i + 1, "total": total, "desc": os.path.basename(input_file)})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"Done. Processed {len(updated)} items")
    print(f"Output file: {output_file}")

    return updated


def augment_dataset_gpt_only(
    dataset_path: str,
    dataset_label: str,
    image_root_prefix: Optional[str],
    weak_cfg: dict,
    output_path: Optional[str] = None,
) -> tuple[int, int]:
    """
    After a single dataset annotations JSON is produced, run traditional model scoring
    and rewrite turns as follows for a NEW output file (do not override the original):
      - HUMAN: set to "<image>\nIs this image real or fake? And by observation of {alias} expert, the blending score is {score|N/A}."
      - GPT: keep original explanation; conditionally append a supporting sentence:
          * real dataset and score <= lo -> append low-score supporting-real sentence
          * fake dataset and score >= hi -> append high-score supporting-fake sentence

    Returns (updated_count, missing_count)
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return (0, 0)

    # Collect absolute paths in order (expect absolute)
    rel_paths = [it.get('image') for it in data if isinstance(it, dict)]
    abs_paths = []
    for p in rel_paths:
        if isinstance(p, str) and os.path.isabs(p):
            abs_paths.append(os.path.normpath(p))
        else:
            raise ValueError("augment_dataset_gpt_only expects absolute image paths in dataset JSON")

    # Prepare provider
    lo = float(((weak_cfg.get('thresholds') or {}).get('lo')) or 0.3)
    hi = float(((weak_cfg.get('thresholds') or {}).get('hi')) or 0.7)
    alias = weak_cfg.get('alias') or 'Blending'
    model_name = weak_cfg.get('model_name') or 'swinv2_base_window16_256'
    weights_path = weak_cfg.get('weights_path') or 'weights/blending_models/best_gf.pth'
    img_size = int(weak_cfg.get('img_size') or 256)
    num_class = int(weak_cfg.get('num_class') or 2)

    provider = get_provider(
        'blending',
        model_name=model_name,
        weights_path=weights_path,
        img_size=img_size,
        num_class=num_class,
        device=None,
    )

    score_map = provider.compute_scores(abs_paths)

    updated, missing = 0, 0
    for it, abs_p in zip(data, abs_paths):
        if not isinstance(it, dict):
            continue
        conv = it.get('conversations') or []
        if not conv:
            continue
        # Find the first HUMAN and GPT turns
        human_idx = None
        gpt_idx = None
        for idx, turn in enumerate(conv):
            if isinstance(turn, dict) and turn.get('from') == 'gpt':
                gpt_idx = idx
                # do not break; prefer the first 'human' before deciding
            if isinstance(turn, dict) and turn.get('from') == 'human' and human_idx is None:
                human_idx = idx
        if gpt_idx is None or human_idx is None:
            continue

        res = score_map.get(abs_p)
        sc = None if res is None else res.score

        # HUMAN: overwrite with question + score
        if sc is None:
            human_text = f"<image>\nIs this image real or fake? And by observation of {alias} expert, the blending score is N/A."
            missing += 1
        else:
            human_text = f"<image>\nIs this image real or fake? And by observation of {alias} expert, the blending score is {float(sc):.3f}."
        conv[human_idx]['value'] = human_text

        # GPT: original explanation plus conditional supporting note
        original_gpt = conv[gpt_idx].get('value') or ''
        note = ''
        if sc is not None:
            if (dataset_label == 'real' and float(sc) <= lo):
                note = f" Additionally, the {alias} model reports a low fake-likelihood score ({float(sc):.3f}), supporting real."
                updated += 1
            elif (dataset_label == 'fake' and float(sc) >= hi):
                note = f" Additionally, the {alias} model reports a high fake-likelihood score ({float(sc):.3f}), supporting fake."
                updated += 1
        conv[gpt_idx]['value'] = original_gpt + note

    # Persist changes to a new dataset file (do not override original)
    out_path = output_path
    if not out_path:
        # Default suffix when not provided
        root, ext = os.path.splitext(dataset_path)
        out_path = root + "_scored" + (ext or ".json")
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return (updated, missing)


## Deprecated: global prefixing was removed; paths come from JSON Description

def main():
    parser = argparse.ArgumentParser(description="Generate deepfake detection explanations (LLaVA-based)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to configuration YAML")
    args = parser.parse_args()

    # Load parameters from config
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ Failed to read config: {e}")
        return

    annotations_cfg = (cfg.get('annotations') or {})
    prompt_file = annotations_cfg.get('prompt_file', 'Prompt/prompt.json')
    model_path = annotations_cfg.get('model_path')
    max_items = annotations_cfg.get('max_items')
    label_override = annotations_cfg.get('label')
    # Prefer annotations.real_files / annotations.fake_files when present
    ann_real_files = list(annotations_cfg.get('real_files') or [])
    ann_fake_files = list(annotations_cfg.get('fake_files') or [])
    # Backward-compatible: plain inputs without labels
    inputs_plain = list(annotations_cfg.get('inputs') or [])
    # Deprecated: image_root_prefix is no longer used for concatenation

    # Build labeled inputs list: [(path, label)]
    data_dir = (cfg.get('paths') or {}).get('data_dir') or 'data'
    use_qa_file_lists = bool(annotations_cfg.get('use_qa_file_lists', False))
    inputs_labeled: list[tuple[str, str]] = []

    if ann_real_files or ann_fake_files:
        # Prefer annotations.real_files / annotations.fake_files
        for f in ann_fake_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'fake'))
        for f in ann_real_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'real'))
    elif use_qa_file_lists or not inputs_plain:
        # Fallback to top-level files.real_files / files.fake_files
        files_cfg = (cfg.get('files') or {})
        real_files = list(files_cfg.get('real_files') or [])
        fake_files = list(files_cfg.get('fake_files') or [])
        if not real_files and not fake_files:
            try:
                from glob import glob
                real_files = [os.path.basename(p) for p in glob(os.path.join(data_dir, 'real*.json'))]
                fake_files = [os.path.basename(p) for p in glob(os.path.join(data_dir, 'fake*.json'))]
            except Exception:
                pass
        for f in fake_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'fake'))
        for f in real_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'real'))
    else:
        # Final fallback: use unlabeled inputs and infer label from filename
        inputs_labeled = [(str(Path(p)), 'auto') for p in inputs_plain]

    # Unified results root: use utils.paths.OUTPUT_ROOT or paths.results_dir (relative to PROJECT_ROOT)
    ensure_core_dirs()
    results_dir_cfg = (cfg.get('paths', {}).get('results_dir') if isinstance(cfg.get('paths'), dict) else None) or str(OUTPUT_ROOT)
    results_root = Path(results_dir_cfg)
    if not results_root.is_absolute():
        results_root = Path(PROJECT_ROOT) / results_root
    annotations_root = results_root / 'annotations'
    runs_root = annotations_root / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime('ann_%Y%m%dT%H%M%SZ')
    run_dir = runs_root / run_id
    datasets_dir = run_dir / 'datasets'
    datasets_dir.mkdir(parents=True, exist_ok=True)
    templates_dir = run_dir / 'templates'
    templates_dir.mkdir(parents=True, exist_ok=True)

    # Progress tracking file：results/annotations/progress.json
    progress_path = annotations_root / 'progress.json'
    tracker = ProgressTracker(progress_path)
    tracker.start()

    # Load prompt template
    prompt_template = load_prompt_template(str(prompt_file))
    print(f"Loaded prompt template: {prompt_template[:100]}...")

    # Process each input file
    merged_outputs = []
    for input_file, label_hint in inputs_labeled:
        # Prefer global label_override; otherwise derive from config/hints
        label = label_override
        if not label:
            if label_hint in ('real', 'fake'):
                label = label_hint
            else:
                # Infer from filename
                lower = os.path.basename(input_file).lower()
                label = "real" if "real" in lower else "fake"

        if os.path.exists(input_file):
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = str(datasets_dir / f"{name_without_ext}_annotations.json")
            template_file = str(templates_dir / f"{name_without_ext}_template.json")

            print(f"\nProcessing file: {input_file} (label: {label})")
            processed = process_json_file(
                input_file,
                output_file,
                prompt_template,
                model_path=model_path,
                max_items=max_items,
                label_override=label,
                image_root_prefix=None,
                progress_tracker=tracker,
                progress_key=name_without_ext,
                template_output_file=template_file,
            )
            # Apply weak-supply augmentation (new file; modify GPT only, keep HUMAN)
            weak_cfg = (cfg.get('weak_supply') or {})
            out_suffix = weak_cfg.get('output_suffix') or '_scored'
            scored_file = str(datasets_dir / f"{name_without_ext}_annotations{out_suffix}.json")
            updated_cnt, missing_cnt = augment_dataset_gpt_only(
                dataset_path=output_file,
                dataset_label=label,
                image_root_prefix=None,
                weak_cfg=weak_cfg,
                output_path=scored_file,
            )
            if updated_cnt or missing_cnt:
                print(f"[WeakSupply][per-dataset] Updated: {updated_cnt}, Missing: {missing_cnt} -> {os.path.basename(scored_file)}")

            # Read augmented file for subsequent merged view
            try:
                with open(scored_file, 'r', encoding='utf-8') as f:
                    processed = json.load(f)
            except Exception:
                pass
            # Build merged view:
            # - human: use augmented prompt (with score line)
            # - gpt: prefix "This image is real/fake." + original answer (may include extra note)
            if processed:
                for it in processed:
                    try:
                        conv = it.get("conversations") or []
                        original_answer = ""
                        human_value = None
                        for turn in conv:
                            if isinstance(turn, dict) and turn.get("from") == "human" and human_value is None:
                                human_value = turn.get("value") or ""
                            if isinstance(turn, dict) and turn.get("from") == "gpt":
                                original_answer = turn.get("value") or ""
                                # do not break; we wanted to see if a prior human exists
                    except Exception:
                        original_answer = ""
                        human_value = None
                    if not human_value:
                        # Fallback: if HUMAN turn is missing, use simplified binary question
                        human_value = f"<image>\n{build_binary_question()}"
                    merged_item = {
                        "id": it.get("id", "0"),  # will be reindexed before final write
                        "image": it.get("image"),
                        "conversations": [
                            {"from": "human", "value": human_value},
                            {"from": "gpt", "value": compose_labeled_response(label, original_answer)},
                        ],
                    }
                    merged_outputs.append(merged_item)
        else:
            print(f"File not found: {input_file}")

    # Shuffle merged results and reassign ids (1..N)
    if merged_outputs:
        random.shuffle(merged_outputs)
    for new_id, item in enumerate(merged_outputs, start=1):
        item["id"] = str(new_id)

    merged_path = run_dir / 'all_annotations_merged.json'
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_outputs, f, ensure_ascii=False, indent=2)

    # Generate run info
    info = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
        'run_id': run_id,
        'model_path': model_path,
        'prompt_template': prompt_template,
        # Note: path prefixes come from each input JSON Description; no global prefix recorded
        'max_items_per_dataset': max_items,
        # Record inputs used (resolved paths if not explicitly configured)
        'inputs': (inputs_plain or [p for p, _ in inputs_labeled]),
        'artefacts': {
            'run_dir': str(run_dir),
            'datasets_dir': str(datasets_dir),
            'merged_annotations': str(merged_path),
        }
    }
    with open(run_dir / 'generation_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Update latest pointer
    latest = annotations_root / 'latest_run.json'
    with open(latest, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # Stop progress tracking
    tracker.stop()

if __name__ == "__main__":
    main()
