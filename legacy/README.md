This folder contains deprecated or experimental scripts kept for reference.

Recommended entrypoints:
- Quick eval (LoRA inference + AUC): `./test.sh`
- Full pipeline (annotate/merge/train/test): `./train.sh --run-train [--train-gpus 0,1]`
- Manual eval: `python -m eval.infer.runner --config eval/configs/infer_config.yaml`
- Compute metrics: `python -m eval.tools.compute_auc`

Scripts kept here may be removed in a future release.
