# Stage 2 — single-image smoke test

First verified real inference through the inherited X2DFD runner on this
machine. One image, no experts, LLaVA-1.5-7B plus the `[ble-diff]` LoRA adapter.

**Result: PASS.** Three earlier attempts failed and are recorded below, because
each one is a real compatibility constraint rather than a transient error.

---

## The run that passed

```bash
.venv\Scripts\python.exe -m tools.run_smoke_test \
  --manifest datasets/raw/data/poc/demo_one.json \
  --experts none --load-4bit --timeout 3600
```

which invoked the inherited runner as:

```bash
.venv\Scripts\python.exe -m eval.infer.runner \
  --config eval/configs/infer_config.yaml \
  --json   datasets/raw/data/poc/demo_one.json \
  --output eval/outputs/smoke_test/demo_one_result.json \
  --experts none
```

with `X2DFD_LOAD_4BIT=1` and `USE_PROGRESS_BAR=0` in the environment.

| Field | Value |
| --- | --- |
| Status | pass (exit code 0 from both wrapper and runner) |
| Wall-clock runtime | 23.271 s (model load + one image) |
| Loading | 4-bit NF4, double quant, fp16 compute |
| Device placement | entire model on GPU 0, no CPU or disk offload |
| Torch-allocated VRAM after load | 4.85 GiB |
| Peak VRAM (nvidia-smi, whole device) | 7400 MiB of 10240 MiB, including ~3 GiB already used by other desktop processes |
| Image | `datasets/raw/images/poc/real_face_01.jpg` |
| Output | `eval/outputs/smoke_test/demo_one_result.json` |
| Summary | `eval/outputs/smoke_test_summary.json` |

Prediction:

| Answer | Label | Real score | Fake score |
| --- | --- | --- | --- |
| "This image is real" | real | 0.6478 | 0.3522 |

Output shape (the schema Stage 3 and Stage 4 will consume):

```json
[
  {
    "id": "1",
    "image": "...\\datasets\\raw\\images\\poc\\real_face_01.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nIs this image real or fake?"},
      {"from": "gpt", "value": "This image is real"},
      {"from": "real score", "value": "0.6478"},
      {"from": "fake score", "value": "0.3522"}
    ]
  }
]
```

The scores are not detector outputs. With `--experts none` the question carries
no expert tail, and `single_image_infer_with_scores` derives them by softmaxing
the model's logits for the `real` and `fake` tokens at one generation step.

The prediction is correct for this image (a public-domain photograph, so the
ground truth is real), but one image says nothing about accuracy. Treat 0.6478
as evidence that the scoring path works, not as a measured result.

---

## Loading path, as it actually is

`utils/lora_inference.py::_get_or_load_model` is the only entry to the model.
It calls the installed `llava.model.builder.load_pretrained_model`, whose
signature is
`(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", ...)`.

Before Stage 2 it passed exactly four positional arguments, so `load_4bit` and
`load_8bit` were always `False` and `device_map` was always `"auto"`: there was
no way to reach the quantised path without editing code. Because the adapter
directory name contains `lora` and a `model_base` is given, the builder takes
its LoRA branch: load base → apply `non_lora_trainables.bin` → attach the
adapter with `PeftModel.from_pretrained` → `merge_and_unload()`.

Two changes were made in this repository (nothing in `site-packages`, nothing in
detector code):

1. `quantisation_from_env()` reads `X2DFD_LOAD_4BIT` / `X2DFD_LOAD_8BIT`, both
   defaulting to off, and the model cache key now includes the mode.
2. `quantisation_kwargs()` builds the `BitsAndBytesConfig` and passes it through
   the loader's existing `**kwargs`, rather than using the loader's own
   `load_4bit` flag. The reason is failure 2 below.

---

## Failures on the way

### 1. Missing `protobuf` — fixed

```text
Inference error: ImportError: The new behaviour of LlamaTokenizer (with
`self.legacy = False`) requires the protobuf library but it was not found
```

`install.sh` never pinned `protobuf`, and `AutoTokenizer.from_pretrained(..., use_fast=False)`
needs it to convert the sentencepiece model. Fixed with
`pip install protobuf` (7.35.1; `pip check` clean).

The failure surfaced 4 seconds into a run rather than at import time, so
`google.protobuf` was added to `CORE_IMPORTS` in `tools/check_environment.py`,
with a regression test in `tests/test_check_environment.py`.

### 2. 4-bit quantised `mm_projector` — worked around

```text
Inference error: RuntimeError: Error(s) in loading state_dict for LlavaLlamaForCausalLM:
  size mismatch for model.mm_projector.0.weight: copying a param with shape
  torch.Size([4096, 1024]) from checkpoint, the shape in current model is
  torch.Size([2097152, 1]).
```

LLaVA's `load_4bit=True` quantises every Linear layer, including the multimodal
projector. `4096 × 1024 ÷ 2 = 2,097,152`, i.e. the projector had become a packed
uint8 tensor, and the builder then tries to load the fp16 projector from
`non_lora_trainables.bin` straight into it. The adapter cannot be applied
without that projector, so skipping the load is not an option.

Worked around by passing a `BitsAndBytesConfig` with
`llm_int8_skip_modules=["mm_projector", "lm_head"]` through the loader's kwargs
while leaving its `load_4bit` flag `False`. Everything else is still NF4. This
uses the installed bitsandbytes integration; no new quantisation code was
written. Regression test in `tests/test_smoke_test_helpers.py`.

### 3. fp16 with CPU offload — a dead end on this machine

Falling back to the stock fp16 path (`device_map="auto"`, model too large for
10 GiB so accelerate offloads part of it to CPU) fails differently:

```text
File "peft/peft_model.py", line 750, in load_adapter
    max_memory = get_balanced_memory(...)
File "accelerate/utils/modeling.py", line 753, in get_balanced_memory
    per_gpu = module_sizes[""] // (num_devices - 1 if low_zero else num_devices)
ZeroDivisionError: integer division or modulo by zero
```

PEFT 0.10.0 re-dispatches the model whenever `hf_device_map` contains `cpu` or
`disk`. It asks accelerate 0.21.0 for a balanced memory map, and
`get_balanced_memory` counts CUDA devices whose **free** memory is above zero
(`get_max_memory` reads `torch.cuda.mem_get_info`). The base model has already
filled the card by that point, so the count is 0 and the division fails.

This is worth recording as a property of the pinned stack: on a single GPU that
the fp16 model cannot fit, the LoRA path cannot use CPU offload at all. Keeping
the whole model on the GPU via 4-bit is not merely faster here, it is the only
configuration that works. The runner also masks it — `lora_infer_conversation_items`
catches the exception and writes it into the answer turn, so the process still
exits 0. That is exactly why the wrapper validates output content rather than
trusting the exit code.

---

## What the wrapper checks

`tools/run_smoke_test.py` treats a run as passing only when all of these hold:

- the manifest parses and every image it names exists with a supported suffix;
- the subprocess exits 0 within the timeout;
- no CUDA out-of-memory signature appears in stdout or stderr;
- the output JSON exists, parses, and is a non-empty list;
- each item has a non-empty `gpt` turn that does not start with `Inference error:`;
- each item has numeric `real score` and `fake score` turns.

Peak VRAM is sampled by polling `nvidia-smi` every 2 s in a background thread,
so the figure covers the whole device, not just this process, and is best-effort.

---

## Reproducing

```bash
.venv\Scripts\python.exe -m pytest -m unit                 # 109 tests, no GPU needed
.venv\Scripts\python.exe -m tools.check_environment --skip-datasets
.venv\Scripts\python.exe -m tools.run_smoke_test --manifest datasets/raw/data/poc/demo_one.json --experts none --load-4bit
```

---

## Remaining caveats

1. **`--experts none` only.** Neither detector has been run through the full
   pipeline yet; both load standalone (see `docs/WEIGHTS_SETUP.md`) but the
   blending and diffusion experts add their own VRAM on top of the 4.85 GiB
   already resident. That is Stage 3.
2. **One image, one prompt.** No accuracy claim is supported.
3. **LoRA merge into 4-bit weights is lossy.** PEFT warns:
   "Merge lora module to 4-bit linear may get different generations due to
   rounding errors." Scores under 4-bit will not match a full-precision run
   exactly, which matters when comparing expert configurations later.
4. **Face cropping.** The test image is a full portrait; upstream X2DFD expects
   DeepfakeBench-style face crops (`docs/TEST_IMAGE_PROVENANCE.md`).
