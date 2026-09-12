"""FF++ c23 source-domain sanity-check helpers (no GPU required).

Lightweight default: deterministic subsample of the official held-out FF++ c23
**test** split (not train), to see whether the inherited X2-DFD setup is broadly
consistent with the paper's strong in-domain FF++ result.

This path is **separate** from the thesis 360-image labelled evaluation
(DeepFakeFace + Celeb-DF-v2).
"""
from __future__ import annotations

import json
import logging
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from eval.labelled_analysis import _roc_auc, classification_metrics
from eval.reproducibility import get_git_provenance, sha256_file, utc_timestamp

LOGGER = logging.getLogger("x2dfd.ffpp_source_domain")

SCRIPT_VERSION = "1.1.0"
# Lightweight sanity defaults (override via CLI for full DeepfakeBench-style protocol).
DEFAULT_FRAMES_PER_VIDEO = 8
FULL_PROTOCOL_FRAMES_PER_VIDEO = 32
DEFAULT_SUBSET_SEED = 4842
DEFAULT_N_REAL_VIDEOS = 50
DEFAULT_N_FAKE_VIDEOS = 50
DATASET_NAME = "FaceForensics++"
COMPRESSION = "c23"
OFFICIAL_SPLIT = "official_test"
VIDEO_AGGREGATION = "mean_frame_fake_score"

# Back-compat alias used by older call sites / tests.
FRAMES_PER_VIDEO = DEFAULT_FRAMES_PER_VIDEO

DEFAULT_ADAPTER = "weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]"
DEFAULT_BASE = "weights/base/llava-v1.5-7b"
DEFAULT_EXPERTS = "blending,diffusion_detector"

MANIPULATION_METHODS = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
)
METHOD_ALIASES = {
    "Deepfakes": "Deepfakes",
    "deepfakes": "Deepfakes",
    "DF": "Deepfakes",
    "Face2Face": "Face2Face",
    "face2face": "Face2Face",
    "F2F": "Face2Face",
    "FaceSwap": "FaceSwap",
    "faceswap": "FaceSwap",
    "FS": "FaceSwap",
    "NeuralTextures": "NeuralTextures",
    "neuraltextures": "NeuralTextures",
    "NT": "NeuralTextures",
    "youtube": "youtube",
    "original": "youtube",
    "real": "youtube",
    "genuine": "youtube",
}

# X2-DFD paper Appendix C.2 Table 9 ("Ours") — comparison metadata only.
PAPER_IN_DOMAIN_AUC = {
    "FF++c23": 0.966,
    "FF++c40": 0.826,
    "FF-DF": 0.999,
    "FF-F2F": 0.972,
    "FF-FS": 0.981,
    "FF-NT": 0.910,
    "AVG": 0.942,
}
PAPER_IN_DOMAIN_SOURCE = (
    "X2-DFD paper Appendix C.2 Table 9 (In-domain results in the FF++ dataset, "
    "AUC as percent → fraction). Row 'Ours'. Not a pass/fail gate. Default "
    "workflow is a subsampled sanity check (100 videos × 8 frames), not a full "
    "test-set reproduction."
)

COMPARISON_MODE = "source_domain_sanity_check"
COMPARISON_RATIONALE = (
    "Inherited public setup: LLaVA-1.5-7B + llava-v1.5-7b-lora-[ble-diff] with "
    "both blending and diffusion specialists (released Baidu/Drive package). "
    "Default run: deterministic 100-video subsample of the official FF++ c23 "
    "held-out test split (50 genuine + 50 manipulated, stratified across the "
    "four methods) with 8 frames/video (~800 planned frames). This is a "
    "source_domain_sanity_check — not an exact reproduction — because we "
    "subsample the test set, use fewer than DeepfakeBench's 32 frames/video, "
    "and Haar-crop faces (tools.make_face_crop) instead of DeepfakeBench "
    "dlib/RetinaFace alignment. Paper Table 9 FF++c23 AUC 0.966 is comparison "
    "metadata only. Use --full-test and --frames-per-video 32 for the fuller "
    "held-out protocol."
)

FACE_PREPROCESS = (
    "tools.make_face_crop.write_face_crop (Haar frontal, size=256, margin=1.3); "
    "approximate DeepfakeBench 256 face crop — not identical to dlib/RetinaFace "
    "landmark alignment used by DeepfakeBench / X2-DFD paper preprocessing"
)


def frame_sampling_description(frames_per_video: int) -> str:
    return (
        f"Deterministic evenly spaced {frames_per_video} frame indices across "
        f"[0, n_frames-1] (inclusive endpoints when n_frames >= {frames_per_video}). "
        f"If n_frames < {frames_per_video}, use all frames. Failed frames are "
        "recorded, not replaced."
    )


FRAME_SAMPLING = frame_sampling_description(DEFAULT_FRAMES_PER_VIDEO)

DEFAULT_SPLIT_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "evaluation"
    / "ffpp_c23_source"
    / "official_splits"
)


class FFPPSourceError(ValueError):
    """FF++ source-domain preparation / analysis failed."""


@dataclass(frozen=True)
class FFPPTestVideo:
    video_id: str
    relative_video_path: str
    relative_frames_dir: Optional[str]
    ground_truth: str  # real | fake
    manipulation: str  # youtube | Deepfakes | Face2Face | FaceSwap | NeuralTextures
    compression: str
    split: str
    pair: Optional[Tuple[str, str]] = None


@dataclass(frozen=True)
class FramePlan:
    frame_id: str
    video_id: str
    video_relpath: str
    ground_truth: str
    manipulation: str
    compression: str
    split: str
    frame_index: int
    sample_slot: int
    total_frames: int


@dataclass(frozen=True)
class SubsetSelection:
    mode: str  # sanity_subset | full_test
    seed: Optional[int]
    n_real_requested: Optional[int]
    n_fake_requested: Optional[int]
    n_real_selected: int
    n_fake_selected: int
    n_fake_by_method: Dict[str, int]
    selected_video_ids: List[str]
    frames_per_video: int
    planned_frames: int


def load_official_pairs(split_path: Path) -> List[Tuple[str, str]]:
    """Load FaceForensics official split JSON (list of [id_a, id_b] pairs)."""

    path = Path(split_path)
    if not path.is_file():
        raise FFPPSourceError(f"official split file missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise FFPPSourceError(f"split JSON must be a non-empty list: {path}")
    pairs: List[Tuple[str, str]] = []
    for row in payload:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise FFPPSourceError(f"invalid pair row in {path}: {row!r}")
        a, b = str(row[0]), str(row[1])
        if not a or not b:
            raise FFPPSourceError(f"empty video id in {path}: {row!r}")
        pairs.append((a, b))
    return pairs


def original_ids_from_pairs(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """Unique original youtube IDs from official pairs (order: first-seen)."""

    out: List[str] = []
    seen = set()
    for a, b in pairs:
        for vid in (a, b):
            if vid not in seen:
                seen.add(vid)
                out.append(vid)
    return out


def manipulated_ids_from_pairs(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """DeepfakeBench-style fake video stems: both pair directions."""

    out: List[str] = []
    seen = set()
    for a, b in pairs:
        for stem in (f"{a}_{b}", f"{b}_{a}"):
            if stem not in seen:
                seen.add(stem)
                out.append(stem)
    return out


def normalise_manipulation(name: str) -> str:
    key = name.strip()
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    raise FFPPSourceError(f"unknown manipulation / source label: {name!r}")


def enumerate_test_videos(
    pairs: Sequence[Tuple[str, str]],
    *,
    compression: str = COMPRESSION,
    split: str = OFFICIAL_SPLIT,
    methods: Sequence[str] = MANIPULATION_METHODS,
) -> List[FFPPTestVideo]:
    """Build the official FF++ test catalogue (reals + four c23 methods)."""

    videos: List[FFPPTestVideo] = []
    for oid in original_ids_from_pairs(pairs):
        videos.append(
            FFPPTestVideo(
                video_id=f"youtube__{oid}",
                relative_video_path=(
                    f"original_sequences/youtube/{compression}/videos/{oid}.mp4"
                ),
                relative_frames_dir=(
                    f"original_sequences/youtube/{compression}/frames/{oid}"
                ),
                ground_truth="real",
                manipulation="youtube",
                compression=compression,
                split=split,
                pair=None,
            )
        )
    for method in methods:
        method_n = normalise_manipulation(method)
        if method_n == "youtube":
            raise FFPPSourceError("youtube is not a manipulated method")
        for stem in manipulated_ids_from_pairs(pairs):
            a, b = stem.split("_", 1)
            videos.append(
                FFPPTestVideo(
                    video_id=f"{method_n}__{stem}",
                    relative_video_path=(
                        f"manipulated_sequences/{method_n}/{compression}/videos/{stem}.mp4"
                    ),
                    relative_frames_dir=(
                        f"manipulated_sequences/{method_n}/{compression}/frames/{stem}"
                    ),
                    ground_truth="fake",
                    manipulation=method_n,
                    compression=compression,
                    split=split,
                    pair=(a, b),
                )
            )
    return videos


def even_quotas(n: int, k: int) -> List[int]:
    """Split ``n`` items across ``k`` bins as evenly as possible (earlier bins get +1)."""

    if k <= 0:
        raise FFPPSourceError("k must be positive")
    if n < 0:
        raise FFPPSourceError("n must be non-negative")
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def select_sanity_subset(
    videos: Sequence[FFPPTestVideo],
    *,
    seed: int = DEFAULT_SUBSET_SEED,
    n_real: int = DEFAULT_N_REAL_VIDEOS,
    n_fake: int = DEFAULT_N_FAKE_VIDEOS,
    methods: Sequence[str] = MANIPULATION_METHODS,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
) -> Tuple[List[FFPPTestVideo], SubsetSelection]:
    """Deterministic 50/50 stratified subsample of the official test catalogue."""

    if n_real < 0 or n_fake < 0:
        raise FFPPSourceError("n_real and n_fake must be non-negative")
    method_names = [normalise_manipulation(m) for m in methods]
    reals = sorted(
        [v for v in videos if v.ground_truth == "real"],
        key=lambda v: v.video_id,
    )
    if n_real > len(reals):
        raise FFPPSourceError(f"requested {n_real} reals but only {len(reals)} available")
    rng = random.Random(seed)
    reals_shuffled = list(reals)
    rng.shuffle(reals_shuffled)
    selected_reals = reals_shuffled[:n_real]

    quotas = even_quotas(n_fake, len(method_names))
    selected_fakes: List[FFPPTestVideo] = []
    n_by_method: Dict[str, int] = {}
    for method, quota in zip(method_names, quotas):
        pool = sorted(
            [v for v in videos if v.manipulation == method],
            key=lambda v: v.video_id,
        )
        if quota > len(pool):
            raise FFPPSourceError(
                f"requested {quota} {method} videos but only {len(pool)} available"
            )
        pool_shuffled = list(pool)
        rng.shuffle(pool_shuffled)
        take = pool_shuffled[:quota]
        selected_fakes.extend(take)
        n_by_method[method] = len(take)

    selected = sorted(
        selected_reals + selected_fakes,
        key=lambda v: (0 if v.ground_truth == "real" else 1, v.manipulation, v.video_id),
    )
    meta = SubsetSelection(
        mode="sanity_subset",
        seed=seed,
        n_real_requested=n_real,
        n_fake_requested=n_fake,
        n_real_selected=len(selected_reals),
        n_fake_selected=len(selected_fakes),
        n_fake_by_method=n_by_method,
        selected_video_ids=[v.video_id for v in selected],
        frames_per_video=frames_per_video,
        planned_frames=len(selected) * frames_per_video,
    )
    return selected, meta


def full_test_selection(
    videos: Sequence[FFPPTestVideo],
    *,
    frames_per_video: int = FULL_PROTOCOL_FRAMES_PER_VIDEO,
) -> Tuple[List[FFPPTestVideo], SubsetSelection]:
    ordered = sorted(
        videos,
        key=lambda v: (0 if v.ground_truth == "real" else 1, v.manipulation, v.video_id),
    )
    n_by_method: Dict[str, int] = {}
    for v in ordered:
        if v.ground_truth == "fake":
            n_by_method[v.manipulation] = n_by_method.get(v.manipulation, 0) + 1
    meta = SubsetSelection(
        mode="full_test",
        seed=None,
        n_real_requested=None,
        n_fake_requested=None,
        n_real_selected=sum(1 for v in ordered if v.ground_truth == "real"),
        n_fake_selected=sum(1 for v in ordered if v.ground_truth == "fake"),
        n_fake_by_method=n_by_method,
        selected_video_ids=[v.video_id for v in ordered],
        frames_per_video=frames_per_video,
        planned_frames=len(ordered) * frames_per_video,
    )
    return ordered, meta


def sample_frame_indices(n_frames: int, n_sample: int = DEFAULT_FRAMES_PER_VIDEO) -> List[int]:
    """Deterministic DeepfakeBench-style evenly spaced frame indices."""

    if n_frames <= 0:
        return []
    if n_sample <= 0:
        raise FFPPSourceError("n_sample must be positive")
    if n_frames <= n_sample:
        return list(range(n_frames))
    if n_sample == 1:
        return [0]
    raw = [int(round(i * (n_frames - 1) / (n_sample - 1))) for i in range(n_sample)]
    out: List[int] = []
    seen = set()
    for idx in raw:
        idx = max(0, min(n_frames - 1, idx))
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    if len(out) < n_sample:
        for idx in range(n_frames):
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
            if len(out) >= n_sample:
                break
    return out[:n_sample]


def probe_video_frame_count(video_path: Path) -> int:
    """Return frame count via OpenCV, or raise FFPPSourceError."""

    import cv2  # type: ignore

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FFPPSourceError(f"cannot open video: {video_path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return max(0, total)
    finally:
        capture.release()


def count_frames_dir(frames_dir: Path) -> int:
    if not frames_dir.is_dir():
        return 0
    return sum(1 for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


def plan_video_frames(
    video: FFPPTestVideo,
    *,
    total_frames: int,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
) -> List[FramePlan]:
    indices = sample_frame_indices(total_frames, frames_per_video)
    return [
        FramePlan(
            frame_id=f"{video.video_id}__f{idx:06d}",
            video_id=video.video_id,
            video_relpath=video.relative_video_path,
            ground_truth=video.ground_truth,
            manipulation=video.manipulation,
            compression=video.compression,
            split=video.split,
            frame_index=idx,
            sample_slot=slot,
            total_frames=total_frames,
        )
        for slot, idx in enumerate(indices)
    ]


def aggregate_video_scores(
    frame_scores: Sequence[Optional[float]],
    *,
    method: str = VIDEO_AGGREGATION,
) -> Optional[float]:
    usable = [float(s) for s in frame_scores if s is not None]
    if not usable:
        return None
    if method != VIDEO_AGGREGATION:
        raise FFPPSourceError(f"unsupported aggregation method: {method}")
    return float(statistics.fmean(usable))


def group_frames_by_video(
    frames: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in frames:
        vid = row.get("video_id")
        if not isinstance(vid, str) or not vid:
            continue
        grouped.setdefault(vid, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: (int(r.get("sample_slot", 0)), int(r.get("frame_index", 0))))
    return grouped


def _pred_label(row: Dict[str, Any]) -> Optional[str]:
    label = row.get("label")
    if label in {"real", "fake"}:
        return label
    score = row.get("fake_score")
    if isinstance(score, (int, float)):
        return "fake" if float(score) >= 0.5 else "real"
    return None


def _confusion_from_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    skipped = 0
    for row in rows:
        if row.get("status") == "failed" or row.get("fake_score") is None:
            skipped += 1
            continue
        gt = row.get("ground_truth")
        pred = _pred_label(row)
        if gt not in {"real", "fake"} or pred is None:
            skipped += 1
            continue
        if gt == "fake" and pred == "fake":
            tp += 1
        elif gt == "real" and pred == "real":
            tn += 1
        elif gt == "real" and pred == "fake":
            fp += 1
        else:
            fn += 1
    metrics = classification_metrics(tp=tp, tn=tn, fp=fp, fn=fn)
    metrics["n_skipped"] = skipped
    return metrics


def _auc_from_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[Optional[float], Optional[str], int, int]:
    scores: List[float] = []
    labels: List[int] = []
    n_failed = 0
    for row in rows:
        score = row.get("fake_score")
        gt = row.get("ground_truth")
        if row.get("status") == "failed" or score is None or gt not in {"real", "fake"}:
            if row.get("status") == "failed" or row.get("failure_reason") or score is None:
                n_failed += 1
            continue
        if not isinstance(score, (int, float)):
            n_failed += 1
            continue
        scores.append(float(score))
        labels.append(1 if gt == "fake" else 0)
    auc, reason = _roc_auc(scores, labels)
    return auc, reason, len(scores), n_failed


def compute_source_domain_metrics(frames: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Frame/video metrics overall and per manipulation family."""

    frame_auc, frame_auc_reason, n_usable, n_failed = _auc_from_rows(frames)
    cls_frame = _confusion_from_rows(frames)

    grouped = group_frames_by_video(list(frames))
    video_rows: List[Dict[str, Any]] = []
    video_score_rows: List[Dict[str, Any]] = []

    for vid, rows in sorted(grouped.items()):
        gt = rows[0].get("ground_truth")
        manip = rows[0].get("manipulation")
        usable_scores = [
            float(r["fake_score"])
            for r in rows
            if r.get("status") != "failed" and isinstance(r.get("fake_score"), (int, float))
        ]
        n_failed_v = sum(
            1 for r in rows if r.get("status") == "failed" or r.get("failure_reason")
        )
        agg = aggregate_video_scores(usable_scores)
        video_label = None
        if agg is not None:
            video_label = "fake" if agg >= 0.5 else "real"
        entry = {
            "video_id": vid,
            "ground_truth": gt,
            "manipulation": manip,
            "n_planned_frames": len(rows),
            "n_usable_frames": len(usable_scores),
            "n_failed_frames": n_failed_v,
            "video_fake_score": agg,
            "video_pred_label": video_label,
            "aggregation": VIDEO_AGGREGATION,
        }
        video_rows.append(entry)
        if agg is not None and gt in {"real", "fake"}:
            video_score_rows.append(
                {
                    "ground_truth": gt,
                    "manipulation": manip,
                    "fake_score": agg,
                    "label": video_label,
                    "status": "scored",
                }
            )

    video_auc, video_auc_reason, n_videos_scored, _ = _auc_from_rows(video_score_rows)
    cls_video = _confusion_from_rows(video_score_rows)

    n_real_videos = sum(1 for v in video_rows if v.get("ground_truth") == "real")
    n_fake_videos = sum(1 for v in video_rows if v.get("ground_truth") == "fake")

    by_manipulation: Dict[str, Any] = {}
    real_frames = [r for r in frames if r.get("manipulation") == "youtube"]
    real_videos = [r for r in video_score_rows if r.get("manipulation") == "youtube"]
    for method in ("youtube",) + MANIPULATION_METHODS:
        if method == "youtube":
            subset_frames = real_frames
            subset_videos = real_videos
            frame_cls = _confusion_from_rows(subset_frames)
            video_cls = _confusion_from_rows(subset_videos)
            by_manipulation[method] = {
                "scope": "genuine_only",
                "n_frames": len(subset_frames),
                "n_videos_scored": len(subset_videos),
                "frame_metrics": frame_cls,
                "video_metrics": video_cls,
                "frame_level_roc_auc": None,
                "video_level_roc_auc": None,
                "note": "AUC undefined for a single class; confusion vs model labels only.",
            }
            continue
        method_frames = [r for r in frames if r.get("manipulation") == method]
        method_videos = [r for r in video_score_rows if r.get("manipulation") == method]
        combined_frames = list(real_frames) + method_frames
        combined_videos = list(real_videos) + method_videos
        f_auc, f_reason, f_n, f_fail = _auc_from_rows(combined_frames)
        v_auc, v_reason, v_n, _ = _auc_from_rows(combined_videos)
        by_manipulation[method] = {
            "scope": "method_fakes_plus_all_reals",
            "n_method_frames": len(method_frames),
            "n_method_videos_scored": len(method_videos),
            "n_combined_usable_frames": f_n,
            "n_combined_failed_or_skipped_frames": f_fail,
            "frame_level_roc_auc": f_auc,
            "frame_level_roc_auc_unavailable_reason": f_reason,
            "video_level_roc_auc": v_auc,
            "video_level_roc_auc_unavailable_reason": v_reason,
            "frame_metrics": _confusion_from_rows(combined_frames),
            "video_metrics": _confusion_from_rows(combined_videos),
            "n_videos_scored_combined": v_n,
        }

    return {
        "n_videos_total": len(grouped),
        "n_videos_scored": n_videos_scored,
        "n_real_videos": n_real_videos,
        "n_fake_videos": n_fake_videos,
        "n_frames_planned": len(frames),
        "n_usable_frames": n_usable,
        "n_failed_frames": n_failed,
        "frame_level_roc_auc": frame_auc,
        "frame_level_roc_auc_unavailable_reason": frame_auc_reason,
        "video_level_roc_auc": video_auc,
        "video_level_roc_auc_unavailable_reason": video_auc_reason,
        "video_aggregation": VIDEO_AGGREGATION,
        "frame_classification": cls_frame,
        "video_classification": cls_video,
        "by_manipulation": by_manipulation,
        "paper_comparison_metadata": {
            "source": PAPER_IN_DOMAIN_SOURCE,
            "not_a_pass_fail_threshold": True,
            "table9_auc": PAPER_IN_DOMAIN_AUC,
            "deltas_vs_table9_FF++c23_frame_auc": (
                None if frame_auc is None else float(frame_auc) - PAPER_IN_DOMAIN_AUC["FF++c23"]
            ),
        },
        "videos": video_rows,
    }


def build_provenance(
    *,
    dataset_root: Path,
    split_path: Path,
    output_dir: Path,
    quantisation: str,
    experts: str,
    adapter_path: str,
    base_path: str,
    load_4bit: bool,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
    subset: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    git = get_git_provenance()
    try:
        split_sha = sha256_file(split_path)
    except OSError:
        split_sha = None
    payload = {
        "generated_at": utc_timestamp(),
        "script": "tools.run_ffpp_source_domain / eval.x2dfd_ffpp_source_domain",
        "script_version": SCRIPT_VERSION,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "comparison_mode": COMPARISON_MODE,
        "comparison_rationale": COMPARISON_RATIONALE,
        "dataset": DATASET_NAME,
        "split": OFFICIAL_SPLIT,
        "compression": COMPRESSION,
        "dataset_root": str(Path(dataset_root).resolve()),
        "official_split_path": str(Path(split_path).resolve()),
        "official_split_sha256": split_sha,
        "output_dir": str(Path(output_dir).resolve()),
        "model_base": base_path,
        "adapter_checkpoint": adapter_path,
        "specialist_configuration": experts,
        "quantisation": quantisation,
        "load_4bit": load_4bit,
        "frames_per_video_target": frames_per_video,
        "frame_sampling_strategy": frame_sampling_description(frames_per_video),
        "face_preprocessing": FACE_PREPROCESS,
        "frame_aggregation_rule": VIDEO_AGGREGATION,
        "subset_selection": subset,
        "paper_comparison_metadata": {
            "source": PAPER_IN_DOMAIN_SOURCE,
            "table9_auc": PAPER_IN_DOMAIN_AUC,
            "not_a_pass_fail_threshold": True,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def build_runner_manifest(
    frames: Sequence[Dict[str, Any]],
    *,
    description_root: Path,
) -> Dict[str, Any]:
    images: List[Dict[str, str]] = []
    for row in frames:
        if row.get("status") == "failed":
            continue
        rel = row.get("crop_rel_path") or row.get("frame_rel_path")
        if not isinstance(rel, str) or not rel:
            continue
        images.append({"image_path": rel.replace("\\", "/")})
    return {
        "Description": str(Path(description_root).resolve()).replace("\\", "/"),
        "images": images,
        "notes": (
            "FF++ c23 source-domain sanity infer input; relative paths "
            "resolve against Description."
        ),
    }


def expected_test_counts(
    pairs: Sequence[Tuple[str, str]],
    *,
    methods: Sequence[str] = MANIPULATION_METHODS,
    frames_per_video: int = FULL_PROTOCOL_FRAMES_PER_VIDEO,
) -> Dict[str, int]:
    n_real = len(original_ids_from_pairs(pairs))
    n_fake_per = len(manipulated_ids_from_pairs(pairs))
    n_methods = len(list(methods))
    n_videos = n_real + n_fake_per * n_methods
    return {
        "n_pairs": len(pairs),
        "n_real_videos": n_real,
        "n_fake_videos_per_method": n_fake_per,
        "n_methods": n_methods,
        "n_videos_total": n_videos,
        "n_planned_frames": n_videos * frames_per_video,
        "frames_per_video": frames_per_video,
    }


def expected_sanity_counts(
    *,
    n_real: int = DEFAULT_N_REAL_VIDEOS,
    n_fake: int = DEFAULT_N_FAKE_VIDEOS,
    frames_per_video: int = DEFAULT_FRAMES_PER_VIDEO,
    methods: Sequence[str] = MANIPULATION_METHODS,
) -> Dict[str, Any]:
    quotas = even_quotas(n_fake, len(list(methods)))
    return {
        "n_videos_total": n_real + n_fake,
        "n_real_videos": n_real,
        "n_fake_videos": n_fake,
        "n_fake_by_method": dict(zip(methods, quotas)),
        "frames_per_video": frames_per_video,
        "n_planned_frames": (n_real + n_fake) * frames_per_video,
        "subset_seed_default": DEFAULT_SUBSET_SEED,
    }


def write_json(path: Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path.resolve()


def subset_selection_to_dict(sel: SubsetSelection) -> Dict[str, Any]:
    return asdict(sel)
