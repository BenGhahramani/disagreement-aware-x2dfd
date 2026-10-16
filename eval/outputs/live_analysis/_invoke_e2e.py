"""One-shot real live-analysis run. Not imported by the dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dashboard.live_analysis import DEFAULT_INFER_CONFIG, analyse_image


def on_progress(event) -> None:
    flag = "OK" if event.ok else "FAIL"
    print(f"[{event.current}/{event.total}] {flag} {event.stage}: {event.message}", flush=True)
    if event.detail:
        print(f"    {event.detail}", flush=True)


def main() -> int:
    source = ROOT / "datasets" / "raw" / "images" / "live" / "mae_carol_jemison.jpg"
    print(f"SOURCE={source}", flush=True)
    print(f"CONFIG={DEFAULT_INFER_CONFIG}", flush=True)
    print("LOAD_4BIT=True (default)", flush=True)

    result = analyse_image(source, progress_callback=on_progress)

    report = result.as_dict()
    out = ROOT / "eval" / "outputs" / "live_analysis" / "_e2e_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT={out}", flush=True)
    print(f"OK={result.ok}", flush=True)
    print(f"WORK_DIR={result.work_dir}", flush=True)
    print(f"CROP={result.crop_path}", flush=True)
    print(f"MANIFEST={result.manifest_path}", flush=True)
    print(f"MATRIX={result.matrix_dir}", flush=True)
    print(f"SUMMARY={result.summary_path}", flush=True)
    if result.view is not None:
        print(f"STATUS={result.view.status.value}", flush=True)
        print(f"RATIONALE={result.view.rationale}", flush=True)
        for card in result.view.cards:
            print(
                f"CARD {card.run_name} label={card.label} real={card.real_score} "
                f"fake={card.fake_score} experts={card.expert_scores} "
                f"runtime_s={card.runtime_s} vram={card.peak_vram_mib} error={card.error}",
                flush=True,
            )
        print(f"CONFLICTS={result.view.evidence_conflicts}", flush=True)
    print(f"ERRORS={result.errors}", flush=True)
    print(f"WARNINGS={result.warnings}", flush=True)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
