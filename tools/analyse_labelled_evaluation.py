#!/usr/bin/env python3
"""Analyse labelled-evaluation aggregate outputs (no inference).

Reads aggregate.json from tools.run_labelled_evaluation and writes metrics,
CSV tables, and simple matplotlib figures. Does not modify raw runner JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from eval.experiment_configs import DEFAULT_CONFIGS, RUN_ORDER, RUN_TITLES
from eval.labelled_analysis import (
    PILOT_WARNING,
    AggregateAnalysisError,
    analyse_aggregate,
    load_and_validate_aggregate,
)
from eval.reproducibility import (
    get_git_provenance,
    interpretation_thresholds,
    sha256_file,
    utc_timestamp,
)

LOGGER = logging.getLogger("x2dfd.analyse_labelled_evaluation")

EXIT_PASS = 0
EXIT_USAGE = 2


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _agreement_crosstab_rows(agreement: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gt, counts in agreement.get("ground_truth_x_evidence_agreement", {}).items():
        for agree, count in counts.items():
            rows.append(
                {
                    "table": "ground_truth_x_evidence_agreement",
                    "row_key": gt,
                    "column_key": agree,
                    "count": count,
                }
            )
    for status, counts in agreement.get(
        "prototype_status_x_evidence_agreement", {}
    ).items():
        for agree, count in counts.items():
            rows.append(
                {
                    "table": "prototype_status_x_evidence_agreement",
                    "row_key": status,
                    "column_key": agree,
                    "count": count,
                }
            )
    return rows


def _annotate_pilot(ax, n_images: int) -> None:
    if n_images <= 4:
        ax.set_title(ax.get_title() + "\n(pipeline validation only; n=4)")


def write_figures(
    analysis: Dict[str, Any],
    images: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> List[str]:
    """Write one matplotlib figure per file (no seaborn, no subplots)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: List[str] = []
    n_images = analysis["n_images"]
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Confusion-like counts per configuration (grouped bars: TP TN FP FN)
    for metrics in analysis["configuration_metrics"]:
        run_name = metrics["run_name"]
        fig, ax = plt.subplots()
        labels = ["TP", "TN", "FP", "FN"]
        values = [metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]]
        ax.bar(labels, values)
        ax.set_ylabel("count")
        ax.set_xlabel("cell")
        title = RUN_TITLES.get(run_name, run_name)
        ax.set_title(f"Confusion counts — {title}")
        _annotate_pilot(ax, n_images)
        path = fig_dir / f"confusion_{run_name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    # Specialist score distributions by ground truth
    for detector, score_key in (
        ("blending", "blending_detector_score"),
        ("diffusion", "diffusion_detector_score"),
    ):
        real_vals: List[float] = []
        fake_vals: List[float] = []
        for image in images:
            cfg = image.get("_configs_by_name", {}).get("blending_diffusion") or {}
            value = cfg.get(score_key)
            if not isinstance(value, (int, float)):
                continue
            if image["ground_truth"] == "real":
                real_vals.append(float(value))
            else:
                fake_vals.append(float(value))
        fig, ax = plt.subplots()
        data = [real_vals, fake_vals]
        ax.boxplot(data, tick_labels=["real GT", "fake GT"])
        ax.set_ylabel(f"{detector} detector score")
        ax.set_title(f"{detector.capitalize()} detector scores by ground truth")
        _annotate_pilot(ax, n_images)
        path = fig_dir / f"specialist_{detector}_by_gt.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    # Model fake-score change relative to baseline
    for comparison in ("blending", "diffusion", "blending_diffusion"):
        deltas = [
            row
            for row in analysis["score_deltas"]
            if row["comparison_run"] == comparison and row["fake_score_delta"] is not None
        ]
        fig, ax = plt.subplots()
        if deltas:
            xs = list(range(len(deltas)))
            ys = [row["fake_score_delta"] for row in deltas]
            ids = [row["image_id"] for row in deltas]
            ax.bar(xs, ys)
            ax.set_xticks(xs)
            ax.set_xticklabels(ids, rotation=45, ha="right")
        ax.axhline(0.0, linewidth=0.8)
        ax.set_ylabel("fake-score delta vs none")
        ax.set_title(f"Observed fake-score change: {comparison} − none")
        _annotate_pilot(ax, n_images)
        path = fig_dir / f"fake_score_delta_{comparison}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    # Evidence-agreement counts
    counts = analysis["agreement"]["counts"]
    fig, ax = plt.subplots()
    keys = ["agreement", "conflict", "insufficient evidence"]
    ax.bar(keys, [counts.get(k, 0) for k in keys])
    ax.set_ylabel("count")
    ax.set_title("Evidence agreement counts")
    _annotate_pilot(ax, n_images)
    path = fig_dir / "evidence_agreement_counts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(str(path))

    return written


def write_analysis_outputs(
    *,
    aggregate_path: Path,
    output_dir: Path,
    analysis: Dict[str, Any],
    images: Sequence[Dict[str, Any]],
    write_plots: bool = True,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    git = get_git_provenance()
    try:
        aggregate_sha = sha256_file(aggregate_path)
    except OSError:
        aggregate_sha = None

    summary = {
        "generated_at": utc_timestamp(),
        "source_aggregate_path": str(aggregate_path.resolve()),
        "source_aggregate_sha256": aggregate_sha,
        "git_commit": git.get("git_commit"),
        "git_dirty": git.get("git_dirty"),
        "canonical_configs": list(DEFAULT_CONFIGS),
        "run_order": list(RUN_ORDER),
        "interpretation_thresholds": interpretation_thresholds(),
        "warning": analysis.get("warning"),
        "pipeline_validation_only": analysis.get("pipeline_validation_only"),
        "analysis": analysis,
    }
    summary_path = output_dir / "analysis_summary.json"
    _write_json(summary_path, summary)

    _write_csv(
        output_dir / "configuration_metrics.csv",
        analysis["configuration_metrics"],
        [
            "run_name",
            "n_usable",
            "n_missing_or_unusable",
            "n_correct",
            "accuracy",
            "tp",
            "tn",
            "fp",
            "fn",
            "sensitivity_fake",
            "recall_fake",
            "specificity_real",
            "balanced_accuracy",
            "precision_fake",
            "f1_fake",
        ],
    )
    _write_csv(
        output_dir / "score_deltas.csv",
        analysis["score_deltas"],
        [
            "image_id",
            "ground_truth",
            "dataset",
            "manipulation",
            "baseline_run",
            "comparison_run",
            "baseline_label",
            "comparison_label",
            "label_changed",
            "transition",
            "baseline_fake_score",
            "comparison_fake_score",
            "fake_score_delta",
        ],
    )
    _write_csv(
        output_dir / "agreement_crosstab.csv",
        _agreement_crosstab_rows(analysis["agreement"]),
        ["table", "row_key", "column_key", "count"],
    )
    _write_csv(
        output_dir / "specialist_summary.csv",
        analysis["specialist_summaries"],
        [
            "detector",
            "score_kind",
            "n_usable",
            "n_real",
            "n_fake",
            "mean_real",
            "median_real",
            "mean_fake",
            "median_fake",
            "min",
            "max",
            "stdev",
            "roc_auc",
            "roc_auc_unavailable_reason",
        ],
    )

    figures: List[str] = []
    if write_plots:
        figures = write_figures(analysis, images, output_dir)

    return {
        "summary_path": str(summary_path.resolve()),
        "figures": figures,
        "output_dir": str(output_dir.resolve()),
    }


def run_analysis(
    *,
    aggregate_path: Path,
    output_dir: Path,
    calibration_min_samples: int = 30,
    write_plots: bool = True,
) -> Dict[str, Any]:
    raw = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    validated = load_and_validate_aggregate(raw)
    analysis = analyse_aggregate(
        validated,
        calibration_min_samples=calibration_min_samples,
    )
    return write_analysis_outputs(
        aggregate_path=Path(aggregate_path),
        output_dir=Path(output_dir),
        analysis=analysis,
        images=validated["images"],
        write_plots=write_plots,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m tools.analyse_labelled_evaluation",
        description="Analyse labelled-evaluation aggregates (no inference).",
    )
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--calibration-min-samples",
        type=int,
        default=30,
        help="minimum n for reporting ECE / full calibration summaries",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip matplotlib figure generation",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        result = run_analysis(
            aggregate_path=args.aggregate,
            output_dir=args.output_dir,
            calibration_min_samples=args.calibration_min_samples,
            write_plots=not args.no_plots,
        )
    except (OSError, json.JSONDecodeError, AggregateAnalysisError) as exc:
        LOGGER.error("%s", exc)
        return EXIT_USAGE
    LOGGER.info("wrote analysis to %s", result["output_dir"])
    if result.get("figures"):
        LOGGER.info("%s figure(s)", len(result["figures"]))
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
