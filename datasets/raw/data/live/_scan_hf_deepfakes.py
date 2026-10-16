"""Scan HF datasets-server for one FF++ Deepfakes c23 frame. Small metadata only."""
from __future__ import annotations

import json
import urllib.request

UA = {"User-Agent": "disagreement-aware-x2dfd-diagnostic/1.0"}
BASE = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=TsienDragon/ffplusplus_c23_frames&config=default&split=test"
)


def fetch_row(offset: int) -> dict:
    url = f"{BASE}&offset={offset}&length=1"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["rows"][0]["row"]


def main() -> None:
    for offset in range(0, 5444, 80):
        row = fetch_row(offset)
        cat = row.get("category")
        label = row.get("label")
        video = row.get("video")
        size = None
        img = row.get("image") or {}
        if isinstance(img, dict):
            size = (img.get("width"), img.get("height"))
        print(f"offset={offset} label={label} category={cat} video={video} size={size}")
        if cat == "Deepfakes" and label == "deepfake":
            print("FOUND")
            print(json.dumps(row, indent=2)[:2000])
            Path = __import__("pathlib").Path
            out = Path("datasets/raw/data/live/_hf_deepfakes_row.json")
            out.write_text(json.dumps({"offset": offset, "row": row}, indent=2) + "\n", encoding="utf-8")
            return
    print("NOT_FOUND")


if __name__ == "__main__":
    main()
