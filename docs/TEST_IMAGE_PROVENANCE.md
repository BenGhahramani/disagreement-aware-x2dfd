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

### Caveat for Stage 2

This is a full upper-body studio portrait: the face occupies only a modest
fraction of the frame. The upstream X2DFD evaluation data is built from
DeepfakeBench preprocessing, which supplies **cropped face** images (see
`README.md` → "Preprocessing: follow DeepfakeBench"), and the bundled dataset
JSONs point at per-frame crops such as
`Tiny_Test/DFDCP/real/1152039_A_001/000.png`.

A full portrait is therefore out of distribution relative to what the LoRA was
trained on. It is adequate for a first mechanical smoke test — does the model
load, run, and emit a parsable verdict — but a face-cropped variant should be
added before any result is treated as evidence about detector behaviour.

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
