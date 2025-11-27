import argparse
import json
import os

from .core import Aligner


def main():
    ap = argparse.ArgumentParser(description="Lightweight Forensics Aligner")
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--models", required=True, help="Comma-separated model folder names under weights_dir")
    ap.add_argument("--paths", required=True, help="Image path | directory | .txt list")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    # gather image paths
    if os.path.isdir(args.paths):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        images = [
            os.path.join(args.paths, f)
            for f in sorted(os.listdir(args.paths))
            if f.lower().endswith(exts)
        ]
    elif args.paths.lower().endswith(".txt") and os.path.isfile(args.paths):
        with open(args.paths) as f:
            images = [ln.strip() for ln in f if ln.strip()]
    else:
        images = [args.paths]

    aligner = Aligner(args.models.split(","), args.weights_dir, device=args.device)
    results = aligner.predict_paths(images)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
    else:
        import pprint
        pprint.pp(results)


if __name__ == "__main__":
    main()

