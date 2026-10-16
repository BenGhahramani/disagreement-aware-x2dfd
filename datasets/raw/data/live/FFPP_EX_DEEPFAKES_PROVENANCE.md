# Known-fake diagnostic sample (FaceForensics++ Deepfakes)

Not a DeepfakeBench-preprocessed 256 crop. Haar crop is required for this run.

## Search result

1. This repo, inherited JSONs, local weights/data, and DeepfakeBench trees do
   not contain a labelled preprocessed FaceForensics++ fake frame. Train
   manifests such as `datasets/raw/data/train/Deepfakes.json` point at
   `FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/...` on a
   missing Linux path (`/data/250010183/Datasets`).
2. A single DeepfakeBench-preprocessed FF++ c23 crop was not available without
   downloading a multi-hundred-MB / multi-GB archive (DeepfakeBench GDrive or
   HuggingFace parquet shards). Those were not started.
3. Selected sample: official FaceForensics++ repository example image.

The FaceForensics++ *blind benchmark* zip was not used (labels hidden).

## Selected image

| Field | Value |
| --- | --- |
| Local file | `datasets/raw/images/live/ffpp_ex_deepfakes.png` |
| Dataset / source | FaceForensics++ official GitHub example (`ondyari/FaceForensics`) |
| Manipulation method | Deepfakes (faceswap autoencoder + Poisson edit) |
| Ground truth | **manipulated / fake** |
| Original path | `images/ex_deepfakes.png` |
| URL | https://github.com/ondyari/FaceForensics/blob/master/images/ex_deepfakes.png |
| Direct download | https://raw.githubusercontent.com/ondyari/FaceForensics/master/images/ex_deepfakes.png |
| Supporting mask | `images/ex_deepfakes_mask.png` → `datasets/raw/images/live/ffpp_ex_deepfakes_mask.png` |
| Mask URL | https://github.com/ondyari/FaceForensics/blob/master/images/ex_deepfakes_mask.png |
| Format | 640 × 480 RGB PNG (236,138 bytes); mask same size (2,966 bytes) |
| Retrieved | 2026-08-08 |

The paired mask is a white rectangle over the face on a black field — the
documented Deepfakes Poisson-edit box. Together with the filename and the
DeepFakes README example section, this is explicit GT=fake, method=Deepfakes.

## Preprocessing vs specialist detectors

DeepfakeBench-style input: dlib/RetinaFace-aligned face, typically **256×256**
PNG, face filling most of the canvas. BlendingDetector then `Resize(256)` +
ImageNet norm. Diffusion Aligner resizes/center-crops toward 224.

This sample is a **full 640×480 video frame** (head + torso + studio backdrop).
It is **not** a DeepfakeBench crop. Blindly skipping Haar would feed mostly
background into the SFDs — the same mismatch the crop tool was written to
avoid.

**Decision for this diagnostic run:** do **not** bypass the live-analysis Haar
crop. Use the same `write_face_crop` stage as the NASA reals (256 JPEG,
margin 1.3). Residual mismatch vs DeepfakeBench alignment may still remain
(Haar vs dlib landmarks); that is part of what this run is meant to expose.
