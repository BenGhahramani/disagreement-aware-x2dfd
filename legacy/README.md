This folder contains deprecated or experimental scripts kept for reference.

Recommended entrypoints:
- Quick eval (LoRA inference): `./test.sh`
- Full pipeline (annotate/merge/train/test): `./train.sh --run-train [--train-gpus 0,1]`
- Manual eval: `python -m eval.infer.runner --config eval/configs/infer_config.yaml`
  
Metrics/AUC computation has been removed from this repository. If you need offline metrics, please compute them externally based on your project needs.

Scripts kept here may be removed in a future release.
