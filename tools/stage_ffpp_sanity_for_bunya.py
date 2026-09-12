#!/usr/bin/env python3
"""After FF++ c23 videos exist: dry-run → prepare → tar.gz for Bunya (no SCP).

Run from repo root after official download completes::

    python tools/stage_ffpp_sanity_for_bunya.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_FFPP = Path(r"C:\Users\Ben\Desktop\UNI\REIT\FaceForensics++")
DEFAULT_PREP = REPO / "datasets" / "evaluation" / "ffpp_c23_source"
DEFAULT_ARCHIVE = Path(r"C:\Users\Ben\Desktop\UNI\REIT\bunya_stage") / "ffpp_c23_source_sanity.tar.gz"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=DEFAULT_FFPP)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_PREP)
    p.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run-only", action="store_true")
    args = p.parse_args()

    py = REPO / ".venv" / "Scripts" / "python.exe"
    if not py.is_file():
        py = Path(sys.executable)

    downloader = args.dataset_root / "download-FaceForensics.py"
    if not downloader.is_file():
        print(
            "Official download-FaceForensics.py not found at:\n"
            f"  {downloader}\n"
            "Place the authorised script there, run run_official_c23_download.ps1,\n"
            "then re-run this staging helper.",
            file=sys.stderr,
        )
        return 2

    videos_ok = any(
        (args.dataset_root / rel).is_dir()
        and any((args.dataset_root / rel).glob("*.mp4"))
        for rel in (
            "original_sequences/youtube/c23/videos",
            "manipulated_sequences/Deepfakes/c23/videos",
        )
    )
    if not videos_ok:
        print(
            "No c23 videos found under the FF++ root. Run the official download first.",
            file=sys.stderr,
        )
        return 2

    dry_cmd = [
        str(py),
        "-m",
        "tools.prepare_ffpp_source_domain",
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(args.output_dir),
        "--dry-run",
    ]
    _run(dry_cmd)

    dry_path = args.output_dir / "preparation_dry_run.json"
    if dry_path.is_file():
        dry = json.loads(dry_path.read_text(encoding="utf-8"))
        sub = dry.get("subset_selection") or {}
        print(
            json.dumps(
                {
                    "n_videos": dry.get("n_videos"),
                    "n_real_videos": dry.get("n_real_videos"),
                    "n_fake_videos": dry.get("n_fake_videos"),
                    "n_fake_by_method": sub.get("n_fake_by_method"),
                    "frames_per_video": sub.get("frames_per_video"),
                    "planned_frames": sub.get("planned_frames"),
                    "seed": sub.get("seed"),
                },
                indent=2,
            )
        )
        if dry.get("n_videos") != 100 or dry.get("n_real_videos") != 50:
            print("Dry-run selection counts unexpected.", file=sys.stderr)
            return 1

    if args.dry_run_only:
        return 0
    if args.skip_prepare:
        return 0

    _run(
        [
            str(py),
            "-m",
            "tools.prepare_ffpp_source_domain",
            "--dataset-root",
            str(args.dataset_root),
            "--output-dir",
            str(args.output_dir),
        ]
    )

    man = args.output_dir / "preparation_manifest.json"
    if not man.is_file():
        print("preparation_manifest.json missing after prepare", file=sys.stderr)
        return 1

    args.archive.parent.mkdir(parents=True, exist_ok=True)
    if args.archive.exists():
        args.archive.unlink()

    # Archive only the prepared subset (crops/frames/manifests/splits), not raw FF++.
    with tarfile.open(args.archive, "w:gz") as tar:
        for name in (
            "preparation_manifest.json",
            "preparation_provenance.json",
            "infer_input.json",
            "official_splits",
            "crops",
            "frames",
        ):
            path = args.output_dir / name
            if path.exists():
                tar.add(path, arcname=f"ffpp_c23_source/{name}")

    print(
        json.dumps(
            {
                "prep_dir": str(args.output_dir.resolve()),
                "prep_bytes": _dir_size(args.output_dir),
                "archive": str(args.archive.resolve()),
                "archive_bytes": args.archive.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
