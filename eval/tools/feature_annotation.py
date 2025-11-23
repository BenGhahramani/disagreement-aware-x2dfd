#!/usr/bin/env python3
"""
根据图片的real/fake标签生成不同的prompt解释

改进：
- 路径前缀仅来自输入 JSON 顶层 Description；不再使用 config.paths.image_root_prefix
- 参数集中由 config.yaml 提供（仅保留 --config）
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
    """加载prompt模板"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
    return prompt_data['prompt']

# generate_explanation 这个函数其实在本脚本中没有被用到，可以删除

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
    """先构造模板，再批量推理填充并保存。

    读取兼容两种输入：
    - 对话风格列表：[{'id','image','conversations': [...]}]
    - Dataset 风格字典：{"Description": "/abs/or/root", "images": [{"image_path": "rel/or/abs"}, ...]}

    行为调整：
    - 若为 Dataset 风格，仅使用 JSON 的 Description 作为根路径前缀；不再回退到 image_root_prefix。
    - 对话风格输入要求 image 已为绝对路径；否则报错提示改用 Dataset 风格并提供 Description。
    - 将所有条目的 image 统一转为绝对路径并写入模板与输出（结果 JSON 的 image 字段为绝对路径）。
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构造模板
    template_items = []
    # 小工具：严格绝对化
    # - Dataset 风格：使用 JSON 顶层 Description 作为 root；若缺失且出现相对路径则报错
    # - 对话风格：必须已是绝对路径
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
            # 对话风格必须提供绝对路径
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
        # 仅使用 JSON 内的 Description；不再回退到 image_root_prefix
        local_root = None
        try:
            desc = data.get('Description') or data.get('description')
            if isinstance(desc, str) and desc.strip():
                local_root = desc.strip()
        except Exception:
            local_root = None
        # 不允许缺少 Description 后仍出现相对路径（在下方绝对化时会抛错）

        items = data['images'][: max_items or len(data['images'])]
        for idx, it in enumerate(items, start=1):
            image_path = it.get('image_path')
            if not image_path:
                continue
            label = label_override or ('real' if 'real' in os.path.basename(input_file).lower() else 'fake')
            personalized_prompt = prompt_template.replace('{label}', label)
            full_prompt = f"<image>\n{personalized_prompt}"
            # 结果中写入绝对路径（必须基于 JSON Description 绝对化或已绝对）
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
        print(f"未知的JSON格式: {type(data)}")
        return []

    if template_output_file:
        with open(template_output_file, 'w', encoding='utf-8') as f:
            json.dump(template_items, f, ensure_ascii=False, indent=2)

    total = len(template_items)
    updated = infer_conversation_items(
        template_items,
        image_root_prefix=None,  # 强制使用绝对路径，不再拼接前缀
        model_path=model_path or "",
        max_new_tokens=512,
    )
    for i in range(total):
        if progress_tracker and progress_key:
            progress_tracker.update(progress_key, {"current": i + 1, "total": total, "desc": os.path.basename(input_file)})

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)

    print(f"处理完成！共处理 {len(updated)} 个条目")
    print(f"输出文件: {output_file}")

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

    # Collect absolute paths in order（要求已为绝对路径）
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


# 已废弃：路径前缀从 JSON 顶层 Description 提供；不再从配置读取

def main():
    parser = argparse.ArgumentParser(description="为图像生成深伪检测解释（基于LLaVA）")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="配置文件路径")
    args = parser.parse_args()

    # 从配置加载参数
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"❌ 读取配置失败: {e}")
        return

    annotations_cfg = (cfg.get('annotations') or {})
    prompt_file = annotations_cfg.get('prompt_file', 'Prompt/prompt.json')
    model_path = annotations_cfg.get('model_path')
    max_items = annotations_cfg.get('max_items')
    label_override = annotations_cfg.get('label')
    # 支持 annotations 中独立的 real_files/fake_files（优先级最高）
    ann_real_files = list(annotations_cfg.get('real_files') or [])
    ann_fake_files = list(annotations_cfg.get('fake_files') or [])
    # 兼容旧配置：inputs 为简单列表（无标签）
    inputs_plain = list(annotations_cfg.get('inputs') or [])
    # 已废弃：不再使用配置中的 image_root_prefix 进行路径拼接

    # 构建注释输入清单，并携带标签：[(path, label)]
    data_dir = (cfg.get('paths') or {}).get('data_dir') or 'data'
    use_qa_file_lists = bool(annotations_cfg.get('use_qa_file_lists', False))
    inputs_labeled: list[tuple[str, str]] = []

    if ann_real_files or ann_fake_files:
        # 优先使用 annotations.real_files / annotations.fake_files
        for f in ann_fake_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'fake'))
        for f in ann_real_files:
            inputs_labeled.append((str(Path(data_dir) / f), 'real'))
    elif use_qa_file_lists or not inputs_plain:
        # 回退到顶层 files.real_files / files.fake_files（QA列表）
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
        # 最后回退：使用无标签 inputs，通过文件名推断
        inputs_labeled = [(str(Path(p)), 'auto') for p in inputs_plain]

    # 结果根目录统一：使用 utils.paths.OUTPUT_ROOT 或配置中的 paths.results_dir（相对路径基于 PROJECT_ROOT）
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

    # 进度追踪文件：results/annotations/progress.json
    progress_path = annotations_root / 'progress.json'
    tracker = ProgressTracker(progress_path)
    tracker.start()

    # 加载prompt模板
    prompt_template = load_prompt_template(str(prompt_file))
    print(f"加载的prompt模板: {prompt_template[:100]}...")

    # 处理每个文件
    merged_outputs = []
    for input_file, label_hint in inputs_labeled:
        # 优先使用全局 label_override；否则按配置/提示确定
        label = label_override
        if not label:
            if label_hint in ('real', 'fake'):
                label = label_hint
            else:
                # 从文件名推断
                lower = os.path.basename(input_file).lower()
                label = "real" if "real" in lower else "fake"

        if os.path.exists(input_file):
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = str(datasets_dir / f"{name_without_ext}_annotations.json")
            template_file = str(templates_dir / f"{name_without_ext}_template.json")

            print(f"\n处理文件: {input_file} (标签: {label})")
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
            # 对该数据集结果进行弱增强（生成新文件；仅改 GPT，HUMAN 不变）
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

            # 读取增强后的新文件用于后续合并视图
            try:
                with open(scored_file, 'r', encoding='utf-8') as f:
                    processed = json.load(f)
            except Exception:
                pass
            # 构建合并视图：
            # - human: 使用增强后的文本（含分数问题行）
            # - gpt: 前缀 "This image is real/fake." + 原回答（可能含附加句）
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
                        # 回退：若意外缺失human，退回到简化二分类问题
                        human_value = f"<image>\n{build_binary_question()}"
                    merged_item = {
                        "id": it.get("id", "0"),  # 将在最终写出前重排
                        "image": it.get("image"),
                        "conversations": [
                            {"from": "human", "value": human_value},
                            {"from": "gpt", "value": compose_labeled_response(label, original_answer)},
                        ],
                    }
                    merged_outputs.append(merged_item)
        else:
            print(f"文件不存在: {input_file}")

    # 随机重排合并结果，然后全局重排 id（1..N）
    if merged_outputs:
        random.shuffle(merged_outputs)
    for new_id, item in enumerate(merged_outputs, start=1):
        item["id"] = str(new_id)

    merged_path = run_dir / 'all_annotations_merged.json'
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_outputs, f, ensure_ascii=False, indent=2)

    # 生成信息记录
    info = {
        'timestamp': datetime.utcnow().isoformat(timespec='seconds'),
        'run_id': run_id,
        'model_path': model_path,
        'prompt_template': prompt_template,
        # 说明：路径前缀由各输入 JSON 的 Description 提供，不再记录全局前缀
        'max_items_per_dataset': max_items,
        # 记录本次运行的输入文件（若未显式配置，则记录解析后的实际文件路径）
        'inputs': (inputs_plain or [p for p, _ in inputs_labeled]),
        'artefacts': {
            'run_dir': str(run_dir),
            'datasets_dir': str(datasets_dir),
            'merged_annotations': str(merged_path),
        }
    }
    with open(run_dir / 'generation_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 更新 latest 指针
    latest = annotations_root / 'latest_run.json'
    with open(latest, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 停止进度追踪
    tracker.stop()

if __name__ == "__main__":
    main()
