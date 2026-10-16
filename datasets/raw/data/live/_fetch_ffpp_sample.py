"""One-off fetch of a labelled FF++ Deepfakes sample. Not part of the demo path."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
IMG_DIR = ROOT / "datasets" / "raw" / "images" / "live"
IMG_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL = {
    "ffpp_ex_deepfakes.png": (
        "https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakes.png"
    ),
    "ffpp_ex_deepfakes_mask.png": (
        "https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakes_mask.png"
    ),
}

UA = {"User-Agent": "disagreement-aware-x2dfd-diagnostic/1.0"}


def fetch(url: str, dest: Path) -> None:
    print(f"GET {url}")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print(f"  wrote {dest} bytes={dest.stat().st_size}")


def inspect(path: Path) -> dict:
    im = Image.open(path)
    info = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "size": list(im.size),
        "mode": im.mode,
        "format": im.format,
    }
    print(f"INFO {path.name}: {info}")
    return info


def main() -> None:
    report = {"official": {}, "hf": None}
    for name, url in OFFICIAL.items():
        dest = IMG_DIR / name
        fetch(url, dest)
        report["official"][name] = inspect(dest)
    out = IMG_DIR / "_ffpp_sample_inspect.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
