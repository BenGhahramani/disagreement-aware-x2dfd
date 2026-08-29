#!/usr/bin/env python3
"""
Fetch the four small pilot images used by datasets/evaluation/pilot.json.

Usage:
    python tools/fetch_pilot_data.py
    python tools/fetch_pilot_data.py --force
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PilotAsset:
    name: str
    relative_path: str
    url: str
    source_note: str


ASSETS = (
    PilotAsset(
        name="NASA Jonny Kim portrait",
        relative_path="datasets/raw/images/poc/real_face_01.jpg",
        url="https://images-assets.nasa.gov/image/jsc2024e052605_alt/jsc2024e052605_alt~orig.jpg",
        source_note="NASA Johnson Space Center, jsc2024e052605_alt",
    ),
    PilotAsset(
        name="NASA Mae Jemison portrait",
        relative_path="datasets/raw/images/live/mae_carol_jemison.jpg",
        url="https://commons.wikimedia.org/wiki/Special:Redirect/file/Mae_Carol_Jemison.jpg",
        source_note="NASA S92-40463 via Wikimedia Commons",
    ),
    PilotAsset(
        name="FaceForensics++ Deepfakes example",
        relative_path="datasets/raw/images/live/ffpp_ex_deepfakes.png",
        url="https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakes.png",
        source_note="ondyari/FaceForensics official repository",
    ),
    PilotAsset(
        name="FaceForensics++ NeuralTextures example",
        relative_path="datasets/raw/images/live/ffpp_ex_neuraltextures.png",
        url="https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_neuraltextures.png",
        source_note="ondyari/FaceForensics official repository",
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "disagreement-aware-x2dfd-pilot-fetch/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise RuntimeError(f"HTTP {status} for {url}")

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type.lower():
                raise RuntimeError(f"Expected an image but received HTML from {url}")

            with temp_path.open("wb") as output:
                shutil.copyfileobj(response, output)

        if temp_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded zero-byte file from {url}")

        temp_path.replace(destination)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def fetch_asset(asset: PilotAsset, root: Path, force: bool) -> bool:
    destination = root / asset.relative_path

    if destination.exists() and not force:
        print(f"[skip] {asset.name}")
        print(f"       path:   {destination}")
        print(f"       bytes:  {destination.stat().st_size}")
        print(f"       sha256: {sha256_file(destination)}")
        return True

    print(f"[fetch] {asset.name}")
    print(f"        source: {asset.source_note}")
    print(f"        url:    {asset.url}")
    print(f"        dest:   {destination}")

    try:
        download(asset.url, destination)
    except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as exc:
        print(f"[error] Failed to fetch {asset.name}: {exc}", file=sys.stderr)
        return False

    print(f"        bytes:  {destination.stat().st_size}")
    print(f"        sha256: {sha256_file(destination)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the four labelled pilot images used by pilot.json."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite files that already exist.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root. Defaults to the parent of this script's tools/ directory.",
    )
    args = parser.parse_args()

    root = args.root.resolve() if args.root else repo_root()

    print(f"Repository root: {root}")
    print("Fetching pilot assets...\n")

    successes = 0
    for asset in ASSETS:
        if fetch_asset(asset, root, args.force):
            successes += 1
        print()

    print(f"Completed: {successes}/{len(ASSETS)} assets available.")

    if successes != len(ASSETS):
        print("One or more downloads failed. No model inference was run.", file=sys.stderr)
        return 1

    print("\nPilot image paths now match datasets/evaluation/pilot.json.")
    print("Record the printed SHA-256 hashes for reproducibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

