import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import CenterCrop, Resize, Compose, InterpolationMode
from torch.utils.data import Dataset, DataLoader

# Optional console progress bar (graceful fallback if not installed)
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Local minimal deps copied into this package
from .processing import make_normalize
from .networks import create_architecture, load_weights


def _build_transform(norm_type: str, patch_size: Union[str, int, Tuple[int, int], None]):
    t: List = []
    if patch_size is None:
        pass
    elif patch_size == "Clip224":
        t += [Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224)]
    elif isinstance(patch_size, (tuple, list)):
        t += [Resize(*patch_size), CenterCrop(patch_size[0])]
    elif isinstance(patch_size, int) and patch_size > 0:
        t += [CenterCrop(patch_size)]
    else:
        raise ValueError(f"Invalid patch_size: {patch_size}")
    t.append(make_normalize(norm_type))
    return Compose(t)


def _logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    # [B] or [B,1] -> sigmoid
    if logits.dim() == 1:
        return torch.sigmoid(logits)
    if logits.dim() == 2:
        c = logits.shape[1]
        if c == 1:
            return torch.sigmoid(logits[:, 0])
        if c == 2:
            return torch.sigmoid(logits[:, 1] - logits[:, 0])
        raise ValueError(f"Unsupported logits with C={c}")
    # [B,C,H,W]
    if logits.dim() == 4:
        b, c, h, w = logits.shape
        if c == 1:
            return torch.sigmoid(logits[:, 0, ...])
        if c == 2:
            return torch.sigmoid(logits[:, 1, ...] - logits[:, 0, ...])
        raise ValueError(f"Unsupported 4D logits with C={c}")
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    arch: str
    norm_type: str
    patch_size: Union[str, int, Tuple[int, int], None]
    weight_path: str


class Aligner:
    """Minimal packaging‑friendly Aligner.

    Example
    -------
    aligner = Aligner(["ours-sync"], "/path/to/weights", device="cuda:0")
    results = aligner.predict_paths(["/path/to/img.jpg"])  # {path: {model: score}}
    """

    def __init__(self, models_list: List[str], weights_dir: str, device: str = "cpu",
                 *,
                 # perf knobs (safe defaults)
                 cudnn_benchmark: bool = True,
                 enable_tf32: bool = True,
                 channels_last: bool = True,
                 use_amp: bool = True,
                 amp_dtype: str = "fp16",  # "fp16" | "bf16"
                 compile_model: bool = False,
                 compile_mode: str = "safe"):
        self.device = torch.device(device)
        self.models: Dict[str, torch.nn.Module] = {}
        self.transforms: Dict[str, Compose] = {}
        self.model_to_transform_key: Dict[str, str] = {}
        self._channels_last = bool(channels_last)
        self._use_amp = bool(use_amp) and (self.device.type == 'cuda')
        self._amp_dtype = torch.float16 if str(amp_dtype).lower() in ("fp16","float16") else torch.bfloat16

        # Backend hints
        try:
            if self.device.type == 'cuda' and bool(cudnn_benchmark):
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            if self.device.type == 'cuda' and bool(enable_tf32):
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass

        specs: List[_ModelSpec] = []
        for mname in models_list:
            cfg = os.path.join(weights_dir, mname, "config.yaml")
            if not os.path.exists(cfg):
                raise FileNotFoundError(f"Missing config: {cfg}")
            import yaml
            with open(cfg) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
            weight_path = os.path.join(weights_dir, mname, data["weights_file"])
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Missing weights: {weight_path}")
            specs.append(_ModelSpec(
                name=mname,
                arch=data["arch"],
                norm_type=data["norm_type"],
                patch_size=data["patch_size"],
                weight_path=weight_path,
            ))

        # Resolve compile mode string once
        def _resolve_compile_mode(mode: str) -> str:
            m = (mode or "").strip().lower()
            # safe: fast compile, best compatibility
            if m in ("safe", "reduce-overhead"):
                return "reduce-overhead"
            # aggressive: longer compile, more autotune opportunities
            if m in ("aggressive", "max-autotune", "max_autotune", "max"):
                return "max-autotune"
            # default fallback
            return "reduce-overhead"

        resolved_compile_mode = _resolve_compile_mode(compile_mode)

        for s in specs:
            tkey = self._tkey(s.patch_size, s.norm_type)
            if tkey not in self.transforms:
                self.transforms[tkey] = _build_transform(s.norm_type, s.patch_size)
            model = load_weights(create_architecture(s.arch), s.weight_path)
            # Optional channels_last for better NHWC kernels
            if self._channels_last:
                try:
                    model = model.to(memory_format=torch.channels_last)
                except Exception:
                    pass
            model = model.to(self.device).eval()
            # Optional torch.compile (PyTorch 2.x)
            if bool(compile_model):
                try:
                    # Prefer resolved mode; gracefully fall back if unsupported
                    try:
                        model = torch.compile(model, mode=resolved_compile_mode)  # type: ignore[attr-defined]
                    except Exception:
                        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[attr-defined]
                except Exception:
                    # torch.compile not available or model not compilable; continue without compile
                    pass
            self.models[s.name] = model
            self.model_to_transform_key[s.name] = tkey

    @staticmethod
    def _tkey(patch_size, norm_type) -> str:
        if patch_size is None:
            return f"none_{norm_type}"
        if patch_size == "Clip224":
            return f"clip224_{norm_type}"
        if isinstance(patch_size, (tuple, list)):
            return f"res{patch_size[0]}_{norm_type}"
        if isinstance(patch_size, int) and patch_size > 0:
            return f"crop{patch_size}_{norm_type}"
        raise ValueError(f"Invalid patch_size: {patch_size}")

    def predict_paths(self, image_paths: List[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        # Gate progress bar via env (default: on)
        use_bar = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
        iterator = (
            tqdm(image_paths, desc="Diffusion Predict", ncols=80, leave=False)
            if (tqdm and use_bar) else image_paths
        )
        with torch.no_grad():
            for p in iterator:
                try:
                    img = Image.open(p).convert("RGB")
                except (UnidentifiedImageError, FileNotFoundError):
                    out[p] = {"error": float("nan")}
                    continue
                out[p] = self._predict_single(img)
        return out

    def _predict_single(self, img: Image.Image) -> Dict[str, float]:
        cache: Dict[str, torch.Tensor] = {}
        for tkey, t in self.transforms.items():
            cache[tkey] = t(img).unsqueeze(0).to(self.device)
        res: Dict[str, float] = {}
        for name, model in self.models.items():
            tkey = self.model_to_transform_key[name]
            logits = model(cache[tkey]).cpu()
            prob = _logits_to_prob(logits)
            score = prob.mean().item() if prob.dim() > 1 else prob.item()
            res[name] = float(score)
        return res

    # New: batched inference over a list of image paths
    def infer(self, image_paths: List[str], batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True,
              *, persistent_workers: bool = True, prefetch_factor: int = 2) -> Dict[str, Dict[str, float]]:
        """Run batched inference for all configured models.

        Returns a mapping {path: {model_name: score}}. Images that fail to load
        receive an entry {"error": nan} to keep output shape stable.
        """
        # Initialize output with placeholders
        out: Dict[str, Dict[str, float]] = {p: {} for p in image_paths}

        class _ImageDS(Dataset):
            def __init__(self, paths: List[str], transform: Compose):
                self.paths = list(paths)
                self.transform = transform
            def __len__(self) -> int:
                return len(self.paths)
            def __getitem__(self, idx: int):
                p = self.paths[idx]
                try:
                    img = Image.open(p).convert("RGB")
                except (UnidentifiedImageError, FileNotFoundError, OSError):
                    return None, p
                try:
                    x = self.transform(img)
                except Exception:
                    return None, p
                return x, p

        def _collate(batch):
            xs = []
            ps: List[str] = []
            for t, p in batch:
                if t is None:
                    # mark failure; score will be missing for all models
                    out[p] = {"error": float("nan")}  # type: ignore[assignment]
                else:
                    xs.append(t)
                    ps.append(p)
            if xs:
                xs = torch.stack(xs, 0)
            else:
                xs = None  # type: ignore[assignment]
            return xs, ps

        with torch.no_grad():
            # Process per-model to reuse its specific transform; cheap since #models is small
            for name, model in self.models.items():
                tkey = self.model_to_transform_key[name]
                transform = self.transforms[tkey]
                ds = _ImageDS(image_paths, transform)
                loader = DataLoader(
                    ds,
                    batch_size=max(1, int(batch_size)),
                    shuffle=False,
                    num_workers=max(0, int(num_workers)),
                    pin_memory=bool(pin_memory),
                    drop_last=False,
                    collate_fn=_collate,
                    persistent_workers=bool(persistent_workers) if int(num_workers) > 0 else False,
                    prefetch_factor=max(2, int(prefetch_factor)) if int(num_workers) > 0 else None,
                )
                # Optional progress bar per-model
                use_bar = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
                it = (
                    tqdm(loader, desc=f"Diffusion[{name}] (bs={batch_size})", ncols=80, leave=False)
                    if (tqdm and use_bar) else loader
                )
                autocast_ctx = torch.cuda.amp.autocast(enabled=self._use_amp, dtype=self._amp_dtype)  # type: ignore[attr-defined]
                for xb, paths in it:
                    if xb is None or not paths:
                        continue
                    xb = xb.to(self.device, non_blocking=True)
                    if self._channels_last:
                        try:
                            xb = xb.contiguous(memory_format=torch.channels_last)
                        except Exception:
                            pass
                    with autocast_ctx:
                        logits = model(xb)
                    logits = logits.float().cpu()
                    prob = _logits_to_prob(logits)
                    # reduce per-sample if multi-dim
                    if prob.dim() > 1:
                        prob = prob.reshape(prob.shape[0], -1).mean(dim=1)
                    scores = prob.tolist()
                    for p, s in zip(paths, scores):
                        # preserve any prior error marker from another pass
                        if isinstance(out.get(p), dict) and "error" in out[p]:
                            continue
                        if p not in out or not isinstance(out[p], dict):
                            out[p] = {}
                        out[p][name] = float(s)
        return out
