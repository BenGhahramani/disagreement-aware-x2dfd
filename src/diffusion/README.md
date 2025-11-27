# Aligner (self-contained)

A minimal, packaging-friendly inference wrapper for the forensics detectors.
This folder includes only the pieces required at runtime:
- aligner/core.py: the `Aligner` class
- aligner/processing.py: image normalization/transforms
- aligner/networks: light network construction + weight loading
- aligner/run.py: tiny CLI for quick tests

Usage
-----
As a module inside this repo:

```
from forensics.aligner import Aligner

aligner = Aligner(["ours-sync"], "/abs/path/to/weights", device="cuda:0")
print(aligner.predict_paths(["/abs/path/to/img.jpg"]))
```

From CLI:

```
python -m forensics.aligner.run \
  --weights_dir /abs/path/to/weights \
  --models ours-sync \
  --paths /abs/path/to/img_or_dir_or_list.txt \
  --device cuda:0 \
  --save /tmp/out.json
```

Weights directory layout:
```
<weights_dir>/<model_name>/config.yaml
<weights_dir>/<model_name>/<weights_file>
```
Where `config.yaml` defines: `arch`, `norm_type`, `patch_size`, `weights_file`.
