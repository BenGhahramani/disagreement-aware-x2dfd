EXPERTS (Traditional Detectors) Integration Guide

Goal
- Support multiple traditional detectors (“experts”) that output P(fake) in [0,1].
- Training and inference share one prompt style (per expert one sentence):
  " And by observation of {alias} expert, the {alias} score is {score}."

What’s Already Included
- Blending expert (src/blending/detector.py) registered as provider "blending".
- Diffusion/forensics expert in two forms:
  - Minimal aligner (provider "aligner" / alias "diffusion") returning per‑model dict.
  - New adapter (src/diffusion/detector.py) registered as provider "diffusion_detector" (alias "diffdet") returning {path: {"score": x}} like blending (recommended for unified style).
- Forensics legacy (src/forensics/*) wrapped by provider "forensics".
- Provider registry and helpers in utils/model_scoring.py:
  - get_provider(name, **kwargs)
  - ForensicsAlignerProvider (each forensics sub-model is one expert)
  - Optional multi-expert helpers: ExpertSpec, ExpertScore, compute_all_scores()
- Prompt helpers (to be used by pipeline/infer): utils/annotation_utils.py (additions planned):
  - format_score(sc), expert_sentence(alias, sc), build_multi_expert_prompt([(alias, sc), ...])

Directory Layout (in this repo)
- src/
  - blending/
    - detector.py              # existing blending detector
  - forensics/
    - __init__.py
    - aligner.py               # loads one or more sub-models by config.yaml
    - networks/
      - __init__.py            # create_architecture/load_weights
      - resnet_mod.py
    - utils/
      - __init__.py
      - processing.py          # preprocessing/normalization

How To Add A New Expert (Two Ways)
1) Wrap a "forensics" sub-model as an expert (recommended for models trained with the forensics code)
   - Prepare weights_dir/<MODEL_NAME>/config.yaml with at least:
     arch: res50 | res50nodown | opencliplinear_... | opencliplinearnext_...
     norm_type: resnet | clip | xception | spec | fft2 | residue3 | npr | cooc
     patch_size: 224 | [H,W] | Clip224 | null
     weights_file: checkpoint.pth
   - Example: weights_dir/ours-sync/config.yaml
       arch: res50
       norm_type: resnet
       patch_size: 224
       weights_file: best.pth
   - Register in your runtime config as one expert (weak_supplies entry):
     - provider: forensics
       alias: OursSync
       weights_dir: /abs/path/to/weights_dir
       model: ours-sync
       thresholds: { lo: 0.35, hi: 0.75 }
       supportive_notes: false
   - The provider creates one Aligner with models_list=[model] and returns P(fake) per image.

2) Add a brand-new provider (custom detector code)
   - Implement a module under src/<your_expert>/detector.py exposing a batching API:
     class YourDetector:
       def __init__(...): ...
       def infer(self, image_paths: List[str], **kwargs) -> Dict[str, Dict[str, float]]:
         # Return: { abs_path: {"score": float in [0,1]} }
   - Register a provider in utils/model_scoring.py:
     class YourProvider(ScoreProvider):
       def __init__(self, ...):
         from src.<your_expert>.detector import YourDetector
         self.detector = YourDetector(...)
       def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
         res = self.detector.infer(image_paths, **kwargs)
         return {p: ScoreResult(path=p, score=(res.get(p) or {}).get("score")) for p in image_paths}
     _REGISTRY["your_expert_key"] = lambda **kw: YourProvider(...)
   - Add one weak_supplies entry per expert you want to expose in prompts.

Configuration (multi-expert)
- weak_supplies is a list; each item equals one expert and becomes one sentence in the prompt.
- Example (two forensics sub-models + blending):
  weak_supplies:
    - { provider: forensics, alias: OursSync,  weights_dir: "/.../forensics/training_Resnet50", model: "ours-sync",  thresholds: {lo: 0.35, hi: 0.75}, supportive_notes: false }
    - { provider: forensics, alias: CorviPlus, weights_dir: "/.../forensics/training_Resnet50", model: "corvi_plus", thresholds: {lo: 0.40, hi: 0.80}, supportive_notes: false }
    - { provider: blending,  alias: Blending,  model_name: "swinv2_base_window16_256", weights_path: "weights/blending_models/best_gf.pth", thresholds: {lo: 0.30, hi: 0.70}, supportive_notes: true }
    - { provider: diffusion_detector, alias: Diffusion, weights_dir: "weights/", model: "ours-sync", thresholds: {lo: 0.30, hi: 0.70}, supportive_notes: false }

Prompt Construction (shared by train/test)
- Base question: "<image>\nIs this image real or fake?"
- Per expert append: expert_sentence(alias, score)
  - " And by observation of {alias} expert, the {alias} score is {N/A|x.xxx}."
- Use utils/annotation_utils.build_multi_expert_prompt to produce the final human question.

Supportive Notes (optional, training-only)
- Per expert thresholds {lo, hi} define confidence.
  - If label==real and score<=lo → append a short supportive sentence for real.
  - If label==fake and score>=hi → append a short supportive sentence for fake.
- Gate with supportive_notes: true/false per expert or a global default.

Programmatic Aggregation (optional helper)
- Describe experts list in code:
  from utils.model_scoring import ExpertSpec, compute_all_scores
  experts = [
    ExpertSpec(provider="forensics", alias="OursSync",  kwargs={"weights_dir": "/...", "model": "ours-sync"}),
    ExpertSpec(provider="blending",  alias="Blending",  kwargs={"weights_path": "weights/blending_models/best_gf.pth"}),
  ]
  score_map = compute_all_scores(experts, abs_paths)  # {path: [ExpertScore, ...]}
  pairs = [(es.alias, es.score) for es in score_map[one_path]]
  human = build_multi_expert_prompt(pairs)

Testing Checklist
- Ensure weights_dir/<model>/config.yaml exists and references a valid weights_file.
- Try a tiny set of 2–3 images: get_provider("forensics", weights_dir=..., model="ours-sync").compute_scores(paths)
- Verify scores are in [0,1]. Missing/failed → score=None → renders as N/A.

Conventions
- alias: short, readable name shown in prompts (e.g., "Blending", "OursSync", "CorviPlus").
- Score semantics: higher means more likely fake (P(fake)). Keep consistent across experts.
- File naming: keep all expert code under src/<expert_name>/ for clarity.
