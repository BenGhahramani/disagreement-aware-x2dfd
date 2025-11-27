import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))

from forensics.aligner import Aligner as NewAligner
from forensics.aligner_min import Aligner as OldAligner


def resolve_paths(json_path, limit=10):
    with open(json_path) as f:
        js = json.load(f)
    root = js.get("Description", "")
    out = []
    for it in js["images"][:limit]:
        p = it.get("image_path") or it.get("path")
        ap = p if os.path.isabs(p) else os.path.join(root, p)
        out.append(ap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--weights_dir", required=True)
    ap.add_argument("--models", default="ours-sync")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    images = resolve_paths(args.json, args.limit)
    models = args.models.split(",")

    new = NewAligner(models, args.weights_dir, device=args.device)
    old = OldAligner(models, args.weights_dir, device=args.device)

    new_res = new.predict_paths(images)
    old_res = old.predict_paths(images)

    deltas = []
    print("\nComparison: path, new_score, old_score, |delta|")
    for p in images:
        ns = list(new_res[p].values())[0]
        oscore = list(old_res[p].values())[0]
        d = abs(ns - oscore)
        deltas.append(d)
        print(f"{os.path.basename(p):>12s}  new={ns:.8f}  old={oscore:.8f}  d={d:.2e}")

    mean_d = sum(deltas)/len(deltas)
    print("\nSummary:")
    print("  mean |delta| =", mean_d)
    print("  max  |delta| =", max(deltas))
    # simple check
    tol = 1e-7
    if max(deltas) > tol:
        raise SystemExit(f"FAIL: max delta {max(deltas)} > tol {tol}")
    print("PASS: all deltas within tolerance", tol)


if __name__ == "__main__":
    main()

