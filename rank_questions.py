#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def normalize_ans(s: str) -> str:
    return (s or "").strip().lower()


def decide_yes_no(item: Dict) -> str:
    """Return 'yes' | 'no' | '' based on item fields.
    Priority: answer text -> fallback to prob_yes/prob_no threshold.
    """
    ans = normalize_ans(item.get("answer", ""))
    if ans.startswith("yes"):
        return "yes"
    if ans.startswith("no"):
        return "no"

    # Fallback via probabilities if available
    if "prob_yes" in item and "prob_no" in item:
        try:
            py = float(item["prob_yes"]) if item["prob_yes"] is not None else 0.0
            pn = float(item["prob_no"]) if item["prob_no"] is not None else 0.0
            if py >= pn:
                return "yes"
            else:
                return "no"
        except Exception:
            pass
    return ""


def compute_rates(
    data: List[Dict],
    questions: List[str],
    positive_on: str,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Compute per-question positive rate.

    positive_on:
      - 'fake' => rate_yes_on_fake (Yes proportion)
      - 'real' => rate_no_on_real (No proportion) by mapping no->positive
    Returns: (rates, counts)
    """
    target_questions = set(q.strip() for q in questions)
    pos = defaultdict(int)
    tot = defaultdict(int)

    for it in data:
        q = it.get("question", "").strip()
        if q not in target_questions:
            continue
        y = decide_yes_no(it)
        if not y:
            continue
        tot[q] += 1
        if positive_on == "fake":
            if y == "yes":
                pos[q] += 1
        elif positive_on == "real":
            if y == "no":
                pos[q] += 1

    rates = {q: (pos[q] / tot[q]) if tot[q] > 0 else 0.0 for q in target_questions}
    counts = {q: tot[q] for q in target_questions}
    return rates, counts


def main():
    ap = argparse.ArgumentParser(description="Rank questions by balanced accuracy from yes/no outputs")
    ap.add_argument("--questions_json", type=str, required=True, help="Path to questions JSON (list of strings)")
    ap.add_argument("--fake_json", type=str, required=True, help="Model outputs on fake set (with question/answer/prob_yes/prob_no)")
    ap.add_argument("--real_json", type=str, required=True, help="Model outputs on real set (with question/answer/prob_yes/prob_no)")
    ap.add_argument("--min_support", type=int, default=1, help="Minimum samples per split to keep a question")
    ap.add_argument("--topk", type=int, default=0, help="If >0, only print top-k questions")
    ap.add_argument("--output_json", type=str, default="", help="Optional: path to save ranked list as JSON")
    args = ap.parse_args()

    questions = load_json(args.questions_json)
    if not isinstance(questions, list):
        raise ValueError("questions_json must be a JSON list of strings")
    questions = [str(q) for q in questions]

    fake_data = load_json(args.fake_json)
    real_data = load_json(args.real_json)

    fake_yes_rate, fake_counts = compute_rates(fake_data, questions, positive_on="fake")
    real_no_rate, real_counts = compute_rates(real_data, questions, positive_on="real")

    ranked = []
    for q in questions:
        n_fake = fake_counts.get(q, 0)
        n_real = real_counts.get(q, 0)
        if n_fake < args.min_support or n_real < args.min_support:
            continue
        fr = fake_yes_rate.get(q, 0.0)
        rr = real_no_rate.get(q, 0.0)
        bal = (fr + rr) / 2.0
        ranked.append({
            "question": q,
            "fake_yes_rate": round(fr, 4),
            "real_no_rate": round(rr, 4),
            "balanced_acc": round(bal, 4),
            "n_fake": n_fake,
            "n_real": n_real,
        })

    ranked.sort(key=lambda x: x["balanced_acc"], reverse=True)

    # Print
    out = ranked if args.topk <= 0 else ranked[: args.topk]
    for i, r in enumerate(out, 1):
        print(
            f"{i:>3}. {r['balanced_acc']:.4f}  | fake_yes={r['fake_yes_rate']:.4f} (n={r['n_fake']})  | real_no={r['real_no_rate']:.4f} (n={r['n_real']})\n    {r['question']}"
        )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(ranked, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

