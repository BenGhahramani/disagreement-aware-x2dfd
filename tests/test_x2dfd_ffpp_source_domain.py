"""Unit tests for FF++ c23 source-domain sanity helpers (no GPU)."""
from __future__ import annotations

from pathlib import Path

import pytest

from eval.x2dfd_ffpp_source_domain import (
    COMPARISON_MODE,
    DEFAULT_FRAMES_PER_VIDEO,
    DEFAULT_SPLIT_DIR,
    DEFAULT_SUBSET_SEED,
    PAPER_IN_DOMAIN_AUC,
    VIDEO_AGGREGATION,
    aggregate_video_scores,
    build_provenance,
    build_runner_manifest,
    compute_source_domain_metrics,
    enumerate_test_videos,
    even_quotas,
    expected_sanity_counts,
    expected_test_counts,
    group_frames_by_video,
    load_official_pairs,
    manipulated_ids_from_pairs,
    normalise_manipulation,
    original_ids_from_pairs,
    sample_frame_indices,
    select_sanity_subset,
)

pytestmark = pytest.mark.unit


def test_load_official_test_split() -> None:
    pairs = load_official_pairs(DEFAULT_SPLIT_DIR / "test.json")
    assert len(pairs) == 70
    originals = original_ids_from_pairs(pairs)
    fakes = manipulated_ids_from_pairs(pairs)
    assert len(originals) == 140
    assert len(fakes) == 140
    counts = expected_test_counts(pairs)
    assert counts["n_videos_total"] == 700
    assert counts["n_planned_frames"] == 700 * 32
    sanity = expected_sanity_counts()
    assert sanity["n_videos_total"] == 100
    assert sanity["n_planned_frames"] == 800
    assert sanity["n_fake_by_method"] == {
        "Deepfakes": 13,
        "Face2Face": 13,
        "FaceSwap": 12,
        "NeuralTextures": 12,
    }


def test_even_quotas() -> None:
    assert even_quotas(50, 4) == [13, 13, 12, 12]
    assert even_quotas(8, 4) == [2, 2, 2, 2]


def test_select_sanity_subset_deterministic() -> None:
    pairs = load_official_pairs(DEFAULT_SPLIT_DIR / "test.json")
    catalogue = enumerate_test_videos(pairs)
    a, meta_a = select_sanity_subset(catalogue, seed=DEFAULT_SUBSET_SEED)
    b, meta_b = select_sanity_subset(catalogue, seed=DEFAULT_SUBSET_SEED)
    assert [v.video_id for v in a] == [v.video_id for v in b]
    assert meta_a.selected_video_ids == meta_b.selected_video_ids
    assert len(a) == 100
    assert meta_a.n_real_selected == 50
    assert meta_a.n_fake_selected == 50
    assert meta_a.mode == "sanity_subset"
    assert meta_a.seed == DEFAULT_SUBSET_SEED
    assert meta_a.frames_per_video == DEFAULT_FRAMES_PER_VIDEO
    assert meta_a.planned_frames == 800
    assert sum(meta_a.n_fake_by_method.values()) == 50
    assert COMPARISON_MODE == "source_domain_sanity_check"
    # Different seed → different selection (with high probability / guaranteed here).
    c, _ = select_sanity_subset(catalogue, seed=DEFAULT_SUBSET_SEED + 1)
    assert [v.video_id for v in a] != [v.video_id for v in c]


def test_enumerate_test_videos_preserves_labels() -> None:
    pairs = [("000", "003"), ("012", "026")]
    videos = enumerate_test_videos(pairs)
    assert len(videos) == 4 + 4 * 4
    reals = [v for v in videos if v.ground_truth == "real"]
    assert len(reals) == 4
    assert all(v.manipulation == "youtube" for v in reals)
    assert all(v.compression == "c23" for v in videos)
    dfs = [v for v in videos if v.manipulation == "Deepfakes"]
    assert len(dfs) == 4


def test_manipulation_mapping() -> None:
    assert normalise_manipulation("DF") == "Deepfakes"
    assert normalise_manipulation("F2F") == "Face2Face"
    assert normalise_manipulation("FS") == "FaceSwap"
    assert normalise_manipulation("NT") == "NeuralTextures"


def test_sample_frame_indices_default_eight() -> None:
    idxs = sample_frame_indices(320, 8)
    assert len(idxs) == 8
    assert idxs[0] == 0
    assert idxs[-1] == 319
    assert sample_frame_indices(100, 8) == sample_frame_indices(100, 8)
    assert sample_frame_indices(5, 8) == list(range(5))


def test_aggregate_and_group() -> None:
    assert aggregate_video_scores([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert VIDEO_AGGREGATION == "mean_frame_fake_score"
    frames = [
        {"video_id": "v2", "sample_slot": 1, "frame_index": 5},
        {"video_id": "v1", "sample_slot": 0, "frame_index": 0},
        {"video_id": "v2", "sample_slot": 0, "frame_index": 1},
    ]
    grouped = group_frames_by_video(frames)
    assert grouped["v2"][0]["sample_slot"] == 0


def test_metrics_auc_failed_frames_and_per_method() -> None:
    frames = [
        {
            "frame_id": "yt_0",
            "video_id": "youtube__000",
            "ground_truth": "real",
            "manipulation": "youtube",
            "sample_slot": 0,
            "frame_index": 0,
            "status": "scored",
            "fake_score": 0.1,
            "label": "real",
        },
        {
            "frame_id": "yt_1",
            "video_id": "youtube__000",
            "ground_truth": "real",
            "manipulation": "youtube",
            "sample_slot": 1,
            "frame_index": 1,
            "status": "scored",
            "fake_score": 0.2,
            "label": "real",
        },
        {
            "frame_id": "ytb_0",
            "video_id": "youtube__003",
            "ground_truth": "real",
            "manipulation": "youtube",
            "sample_slot": 0,
            "frame_index": 0,
            "status": "scored",
            "fake_score": 0.15,
            "label": "real",
        },
        {
            "frame_id": "df_0",
            "video_id": "Deepfakes__000_003",
            "ground_truth": "fake",
            "manipulation": "Deepfakes",
            "sample_slot": 0,
            "frame_index": 0,
            "status": "scored",
            "fake_score": 0.9,
            "label": "fake",
        },
        {
            "frame_id": "df_2",
            "video_id": "Deepfakes__003_000",
            "ground_truth": "fake",
            "manipulation": "Deepfakes",
            "sample_slot": 0,
            "frame_index": 0,
            "status": "scored",
            "fake_score": 0.88,
            "label": "fake",
        },
        {
            "frame_id": "df_1",
            "video_id": "Deepfakes__000_003",
            "ground_truth": "fake",
            "manipulation": "Deepfakes",
            "sample_slot": 1,
            "frame_index": 1,
            "status": "failed",
            "failure_reason": "no_face",
            "fake_score": None,
        },
        {
            "frame_id": "f2f_0",
            "video_id": "Face2Face__000_003",
            "ground_truth": "fake",
            "manipulation": "Face2Face",
            "sample_slot": 0,
            "frame_index": 0,
            "status": "scored",
            "fake_score": 0.85,
            "label": "fake",
        },
    ]
    metrics = compute_source_domain_metrics(frames)
    assert metrics["n_failed_frames"] == 1
    assert metrics["n_usable_frames"] == 6
    assert metrics["frame_level_roc_auc"] == pytest.approx(1.0)
    assert metrics["video_level_roc_auc"] == pytest.approx(1.0)
    assert metrics["frame_classification"]["accuracy"] == pytest.approx(1.0)
    assert metrics["paper_comparison_metadata"]["table9_auc"]["FF++c23"] == PAPER_IN_DOMAIN_AUC[
        "FF++c23"
    ]
    assert metrics["by_manipulation"]["Deepfakes"]["frame_level_roc_auc"] == pytest.approx(1.0)


def test_runner_manifest_and_provenance(tmp_path: Path) -> None:
    frames = [
        {"status": "cropped", "crop_rel_path": "crops/a.jpg", "frame_id": "a"},
        {"status": "failed", "failure_reason": "no_face", "frame_id": "b"},
    ]
    man = build_runner_manifest(frames, description_root=tmp_path)
    assert len(man["images"]) == 1
    split = DEFAULT_SPLIT_DIR / "test.json"
    prov = build_provenance(
        dataset_root=tmp_path,
        split_path=split,
        output_dir=tmp_path / "out",
        quantisation="fp16",
        experts="blending,diffusion_detector",
        adapter_path="weights/checkpoints/ckpt/llava-v1.5-7b-lora-[ble-diff]",
        base_path="weights/base/llava-v1.5-7b",
        load_4bit=False,
        frames_per_video=8,
        subset={"mode": "sanity_subset", "seed": 4842, "planned_frames": 800},
    )
    assert prov["comparison_mode"] == "source_domain_sanity_check"
    assert prov["frames_per_video_target"] == 8
    assert prov["subset_selection"]["planned_frames"] == 800
    assert "0.904" not in __import__("json").dumps(prov)
