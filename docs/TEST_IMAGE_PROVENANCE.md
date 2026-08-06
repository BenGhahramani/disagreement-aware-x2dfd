# Test image provenance

Provenance for the single image used to smoke-test the inherited X2DFD
pipeline. The image itself lives under `datasets/raw/images/`, which is
gitignored, so this file is the tracked record of where it came from.

## `datasets/raw/images/poc/real_face_01.jpg`

| Field | Value |
| --- | --- |
| Subject | Official NASA portrait of astronaut Jonny Kim |
| Source page | https://commons.wikimedia.org/wiki/File:Jsc2024e052605_alt_(Aug._6,_2024)_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg |
| Direct file URL | https://upload.wikimedia.org/wikipedia/commons/e/e9/Jsc2024e052605_alt_%28Aug._6%2C_2024%29_---_Official_portrait_of_NASA_astronaut_Jonny_Kim.jpg |
| Licence | Public domain (`LicenseShortName` and `UsageTerms` both "Public domain" via the Commons API) |
| Rights basis | Work of NASA, a US federal agency; NASA imagery is generally not subject to copyright |
| Credit | NASA Johnson Space Center / Josh Valcarcel — NASA photo ID `jsc2024e052605_alt` |
| Date taken | 6 August 2024 |
| Downloaded | 2026-08-06 |
| Format / size | JPEG, 5202 × 6502, RGB, 2,927,608 bytes |

The downloaded byte count matches the size reported by the Commons API exactly,
so the file is complete rather than truncated. Verified with Pillow:
`Image.verify()` passed and the image loads as `JPEG (5202, 6502) RGB`.

Suitability against the stated requirements: identifiable human face, clear and
front-facing, unobstructed, high resolution, JPEG, public domain, an official
published portrait rather than a private or sensitive photograph.

### Caveat for Stage 2 (addressed in Stage 3)

This is a full upper-body studio portrait: the face occupies only a modest
fraction of the frame. The upstream X2DFD evaluation data is built from
DeepfakeBench preprocessing, which supplies **cropped face** images (see
`README.md` → "Preprocessing: follow DeepfakeBench"), and the bundled dataset
JSONs point at per-frame crops such as
`Tiny_Test/DFDCP/real/1152039_A_001/000.png`.

A full portrait is therefore out of distribution relative to what the LoRA was
trained on. It was adequate for the Stage 2 mechanical smoke test — does the
model load, run, and emit a parsable verdict — but the expert matrix runs on the
face crop below instead.

## `datasets/raw/images/poc/real_face_01_crop.jpg`

A face crop derived from the portrait above. The original is untouched
(2,927,608 bytes, unchanged).

| Field | Value |
| --- | --- |
| Derived from | `real_face_01.jpg` (5202 × 6502) |
| Method | OpenCV Haar frontal-face cascade, largest detection, squared and margined |
| Detection box (source px) | x=2343, y=1054, w=946, h=946 |
| Crop box (source px) | x=2201, y=912, w=1230, h=1230 |
| Output | JPEG, 256 × 256, RGB, quality 95, 24,630 bytes |
| SHA-256 | `e8bd4ea46bd95e8cf7ac82eaabf0a8ab4adb55506d7ea8b7ca9420cc75142fd8` |
| Created | 2026-08-06 |

Produced with `tools/make_face_crop.py`:

```bash
.venv\Scripts\python.exe -m tools.make_face_crop \
  --image  datasets/raw/images/poc/real_face_01.jpg \
  --output datasets/raw/images/poc/real_face_01_crop.jpg
```

The recipe, at its default parameters:

1. Downscale a copy to 1024 px on the long side (`INTER_AREA`) for detection
   only, so detection cost does not scale with source resolution.
2. Greyscale, `cv2.equalizeHist`, then
   `haarcascade_frontalface_default.xml` with `scaleFactor=1.1`,
   `minNeighbors=5`, `minSize=(48, 48)`.
3. Take the largest detection and map it back to full-resolution coordinates
   (here the detection ran at ratio 1024/6502 ≈ 0.1575).
4. Expand to a square of `1.3 ×` the longer detection side, centred on the
   detection and clamped to the image bounds.
5. Resize to 256 × 256 with `INTER_AREA` and write JPEG at quality 95.

**Determinism:** Haar cascades involve no randomness, and every parameter above
is fixed, so re-running reproduces the file byte-for-byte. Verified by running
the tool a second time to a temporary path and comparing SHA-256 — identical.

**Why 256 × 256:** it matches both detectors without further distortion.
`src/blending/detector.py` resizes its input to `(img_size, img_size)` with
`img_size: 256` from `eval/configs/infer_config.yaml`, and the diffusion expert
applies `CenterCrop(224)` with no resize, which keeps the central 224 px of the
face rather than an arbitrary 224 px patch of a large image.

**Still a caveat:** this approximates DeepfakeBench's preprocessing but is not
identical to it. DeepfakeBench uses a landmark-based aligner (dlib/RetinaFace)
with its own margin convention, so the framing here will differ somewhat from
the crops the LoRA was actually trained on.

### Manifest for the crop

`datasets/raw/data/poc/demo_one_crop.json`, same absolute-`Description`
convention as below:

```json
{
  "Description": "C:/Users/Ben/Desktop/UNI/REIT/root/disagreement-aware-x2dfd/datasets/raw/images/poc",
  "images": [{ "image_path": "real_face_01_crop.jpg", "label": "real" }]
}
```

## Manifest

`datasets/raw/data/poc/demo_one.json`:

```json
{
  "Description": "C:/Users/Ben/Desktop/UNI/REIT/root/disagreement-aware-x2dfd/datasets/raw/images/poc",
  "images": [{ "image_path": "real_face_01.jpg" }]
}
```

`Description` is an **absolute** path on purpose.
`eval/infer/runner.py` resolves images via
`resolve_abs_paths(imgs, root_prefix=Description)`, which is
`os.path.normpath(os.path.join(root, rel))` with no `abspath` call, so a
relative `Description` would resolve against whatever the current working
directory happens to be. Absolute also matches upstream convention — the
shipped dataset JSONs use `/data/250010183/Datasets`.

The trade-off is that this manifest is machine-specific and will need its
`Description` edited on any other machine (or replaced with a
`${X2DFD_DATASETS}`-style variable, which the runner expands).

Verified 2026-08-06:

```text
$ resolve_abs_paths(...)  ->  ...\datasets\raw\images\poc\real_face_01.jpg | exists: True
$ tools.check_environment.check_dataset_inputs(...)
PASS datasets.demo_one.json | demo_one.json: 1 sampled image(s) exist
```
