"""
Unified model scoring interfaces and providers.

This module exposes a small registry to support multiple traditional
detectors. It currently includes a 'blending' provider backed by
src.blending.detector.BlendingDetector, and can be extended to other
models later by adding new providers and registering them.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from pathlib import Path

# Optional progress bar for multi-expert aggregation
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# Ensure repo root on sys.path for `src` imports when executed as a script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Providers may be optional; import lazily in providers


@dataclass
class ScoreResult:
    path: str
    score: Optional[float]


class ScoreProvider:
    def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
        raise NotImplementedError


class BlendingProvider(ScoreProvider):
    def __init__(self, model_name: str, weights_path: str, img_size: int = 256, num_class: int = 2, device: Optional[str] = None,
                 batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True) -> None:
        from src.blending.detector import BlendingDetector  # local import
        import torch
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.detector = BlendingDetector(
            model_name=model_name,
            weights_path=weights_path,
            img_size=img_size,
            num_class=num_class,
            device=device,
        )

    def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
        # Allow call-time overrides, else use defaults from __init__
        bs = int(kwargs.get('batch_size', self.batch_size))
        nw = int(kwargs.get('num_workers', self.num_workers))
        pm = bool(kwargs.get('pin_memory', self.pin_memory))
        results = self.detector.infer(image_paths, batch_size=bs, num_workers=nw, pin_memory=pm)
        out: Dict[str, ScoreResult] = {}
        for p in image_paths:
            entry = results.get(p)
            score = None if entry is None else entry.get("score")
            out[p] = ScoreResult(path=p, score=score if score is None else float(score))
        return out


class DiffusionDetectorProvider(ScoreProvider):
    """Provider wrapper around src.diffusion.detector.DiffusionDetector.

    Mirrors the BlendingProvider batching API and returns ScoreResult.
    """

    def __init__(self, *, weights_dir: str, model: str, device: Optional[str] = None,
                 batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True,
                 compile_model: bool = False, compile_mode: str = "safe") -> None:
        import torch
        from src.diffusion.detector import DiffusionDetector  # local import

        if not model or not isinstance(model, str):
            raise ValueError("DiffusionDetectorProvider: 'model' must be a non-empty string.")
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.detector = DiffusionDetector(
            weights_dir=weights_dir,
            model=model,
            device=device,
            compile_model=bool(compile_model),
            compile_mode=str(compile_mode),
        )

    def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
        bs = int(kwargs.get('batch_size', self.batch_size))
        # Guard against overly large batches on high‑res inputs
        if bs > 64:
            bs = 64
        nw = int(kwargs.get('num_workers', self.num_workers))
        pm = bool(kwargs.get('pin_memory', self.pin_memory))
        results = self.detector.infer(
            image_paths,
            batch_size=bs,
            num_workers=nw,
            pin_memory=pm,
            persistent_workers=False,
            prefetch_factor=2,
        )
        out: Dict[str, ScoreResult] = {}
        for p in image_paths:
            entry = results.get(p)
            sc = None
            if isinstance(entry, dict):
                v = entry.get('score')
                if isinstance(v, (int, float)):
                    sc = float(v)
            out[p] = ScoreResult(path=p, score=sc)
        return out

class ForensicsAlignerProvider(ScoreProvider):
    """Provider wrapper around the legacy src.forensics.aligner.Aligner.

    Note: Some repositories may not include src/forensics; in such cases use
    the new 'aligner' provider backed by src/aligner (see AlignerProvider).

    This treats one 'forensics' sub-model (folder under weights_dir) as one expert.
    The sub-model folder must contain a config.yaml with keys like arch/norm_type/patch_size/weights_file.

    Args:
        weights_dir: Root dir containing per-model subfolders.
        model: Name of the subfolder (one per expert), e.g. 'ours-sync'.
        device: torch device; auto-detect if None.
    """

    def __init__(self, *, weights_dir: str, model: str, device: Optional[str] = None) -> None:
        import torch  # local import to avoid hard dep on torch at import time
        # defer import to avoid hard dependency when not used
        from src.forensics.aligner import Aligner  # type: ignore

        if not model or not isinstance(model, str):
            raise ValueError("ForensicsAlignerProvider: 'model' must be a non-empty string (subfolder name under weights_dir).")

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.weights_dir = str(weights_dir)
        self.model_name = str(model)
        self.device = device

        # Initialize a single-model aligner; it will read config from {weights_dir}/{model}/config.yaml
        self._aligner = Aligner(models_list=[self.model_name], weights_dir=self.weights_dir, device=self.device)

    def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
        # legacy forensics wrapper expects method name 'predict_paths' or 'infer'; support both
        if hasattr(self._aligner, 'predict_paths'):
            results = self._aligner.predict_paths(image_paths)  # type: ignore[attr-defined]
        else:
            results = self._aligner.infer(image_paths)  # type: ignore[attr-defined]
        out: Dict[str, ScoreResult] = {}
        for p in image_paths:
            sc: Optional[float] = None
            entry = results.get(p)
            if isinstance(entry, dict):
                # prefer explicit {'score': x}; else allow {model_name: x}
                if 'score' in entry and isinstance(entry['score'], (int, float)):
                    sc = float(entry['score'])
                elif self.model_name in entry and isinstance(entry[self.model_name], (int, float)):
                    sc = float(entry[self.model_name])
            out[p] = ScoreResult(path=p, score=sc)
        return out


class AlignerProvider(ScoreProvider):
    """Provider wrapper around the in-repo src.aligner.core.Aligner.

    This is the recommended minimal diffusion/forensics expert integration
    for this repository. Each sub-folder under weights_dir represents one
    expert (model), identified by `model`.

    Args:
        weights_dir: Root directory containing <model>/config.yaml and weights.
        model:       Subfolder name for one expert (e.g., 'ours-sync').
        device:      torch device string; auto-detected if None.
        lazy:        When True, delay heavy initialization until first call.
    """

    def __init__(self, *, weights_dir: str, model: str, device: Optional[str] = None, lazy: bool = True,
                 batch_size: int = 64, num_workers: int = 4, pin_memory: bool = True,
                 compile_model: bool = False, compile_mode: str = "safe") -> None:
        import torch  # local import
        if not model or not isinstance(model, str):
            raise ValueError("AlignerProvider: 'model' must be a non-empty string (subfolder name under weights_dir).")

        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.weights_dir = str(weights_dir)
        self.model_name = str(model)
        self.device = device
        self._lazy = bool(lazy)
        # batch preferences (used when aligner exposes infer())
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self._aligner = None  # type: ignore[var-annotated]
        self.compile_model = bool(compile_model)
        self.compile_mode = str(compile_mode)

        # Eager init when requested
        if not self._lazy:
            self._ensure_aligner()

    def _ensure_aligner(self) -> None:
        if self._aligner is None:
            # Prefer new location src.diffusion; fallback to src.aligner for backward-compat
            try:
                from src.diffusion.core import Aligner  # type: ignore
                # Newer in-repo aligner supports compile knobs
                self._aligner = Aligner(
                    models_list=[self.model_name],
                    weights_dir=self.weights_dir,
                    device=self.device,
                    compile_model=self.compile_model,
                    compile_mode=self.compile_mode,
                )
            except Exception:
                # Fallback for older layout without compile knobs
                from src.aligner.core import Aligner  # type: ignore
                self._aligner = Aligner(models_list=[self.model_name], weights_dir=self.weights_dir, device=self.device)

    def compute_scores(self, image_paths: List[str], **kwargs) -> Mapping[str, ScoreResult]:
        self._ensure_aligner()
        # mypy: _aligner is ensured non-None
        # Prefer batched infer() when available; fallback to per-image predict_paths
        if hasattr(self._aligner, 'infer'):
            try:
                results = self._aligner.infer(image_paths, batch_size=self.batch_size,
                                              num_workers=self.num_workers, pin_memory=self.pin_memory)  # type: ignore[union-attr]
            except Exception:
                results = self._aligner.predict_paths(image_paths)  # type: ignore[union-attr]
        else:
            results = self._aligner.predict_paths(image_paths)  # type: ignore[union-attr]
        out: Dict[str, ScoreResult] = {}
        for p in image_paths:
            sc: Optional[float] = None
            entry = results.get(p)
            if isinstance(entry, dict):
                # prefer explicit {'score': x}; else read by model name
                if 'score' in entry and isinstance(entry['score'], (int, float)):
                    sc = float(entry['score'])
                elif self.model_name in entry and isinstance(entry[self.model_name], (int, float)):
                    sc = float(entry[self.model_name])
            out[p] = ScoreResult(path=p, score=sc)
        return out


ProviderFactory = Callable[..., ScoreProvider]


_REGISTRY: Dict[str, ProviderFactory] = {
    'blending': lambda **kw: BlendingProvider(
        model_name=kw.get('model_name', 'swinv2_base_window16_256'),
        weights_path=kw.get('weights_path', 'weights/blending_models/best_gf.pth'),
        img_size=int(kw.get('img_size', 256)),
        num_class=int(kw.get('num_class', 2)),
        device=kw.get('device'),
        # Default batch size bumped from 64 -> 256 per user request; callers can still override.
        batch_size=int(kw.get('batch_size', 256)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
    ),
    # Treat each forensics sub-model as one expert; requires in-repo src/forensics/*.
    'forensics': lambda **kw: ForensicsAlignerProvider(
        weights_dir=kw.get('weights_dir', ''),
        model=kw.get('model', ''),
        device=kw.get('device'),
    ),
    # New minimal diffusion/forensics expert backed by src/diffusion (or src/aligner fallback)
    'aligner': lambda **kw: AlignerProvider(
        weights_dir=kw.get('weights_dir', ''),
        model=kw.get('model', ''),
        device=kw.get('device'),
        lazy=bool(kw.get('lazy', True)),
        # Default batch size bumped from 64 -> 256 so diffusion/aligner experts use larger batches
        # when the pipeline does not explicitly pass batching knobs (e.g., multi-expert path).
        batch_size=int(kw.get('batch_size', 256)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
        compile_model=bool(kw.get('compile_model', False)),
        compile_mode=str(kw.get('compile_mode', 'safe')),
    ),
    # Alias: expose the same provider under 'diffusion'
    'diffusion': lambda **kw: AlignerProvider(
        weights_dir=kw.get('weights_dir', ''),
        model=kw.get('model', ''),
        device=kw.get('device'),
        lazy=bool(kw.get('lazy', True)),
        # Keep 'diffusion' alias consistent with 'aligner' default: 256
        batch_size=int(kw.get('batch_size', 256)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
        compile_model=bool(kw.get('compile_model', False)),
        compile_mode=str(kw.get('compile_mode', 'safe')),
    ),
    # New adapter provider that returns {'score': x} like blending
    'diffusion_detector': lambda **kw: DiffusionDetectorProvider(
        weights_dir=kw.get('weights_dir', ''),
        model=kw.get('model', ''),
        device=kw.get('device'),
        batch_size=int(kw.get('batch_size', 256)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
        compile_model=bool(kw.get('compile_model', False)),
        compile_mode=str(kw.get('compile_mode', 'safe')),
    ),
    'diffdet': lambda **kw: DiffusionDetectorProvider(
        weights_dir=kw.get('weights_dir', ''),
        model=kw.get('model', ''),
        device=kw.get('device'),
        batch_size=int(kw.get('batch_size', 256)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
        compile_model=bool(kw.get('compile_model', False)),
        compile_mode=str(kw.get('compile_mode', 'safe')),
    ),
}


def get_provider(name: str, **kwargs) -> ScoreProvider:
    name_l = (name or '').strip().lower()
    if name_l not in _REGISTRY:
        raise KeyError(f"Unknown score provider: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name_l](**kwargs)


# ---- Multi-expert aggregation helpers (optional; non-breaking) ----

@dataclass
class ExpertSpec:
    provider: str
    alias: str
    kwargs: Dict[str, object]


@dataclass
class ExpertScore:
    provider: str
    alias: str
    score: Optional[float]


def compute_all_scores(experts: List[ExpertSpec], image_paths: List[str]) -> Dict[str, List[ExpertScore]]:
    """Compute scores for multiple experts and return mapping per image.

    The order of ExpertScore entries per image follows the input experts order,
    which is important for deterministic prompt construction.
    """
    # Initialize output with empty lists to preserve keys for images without results
    out: Dict[str, List[ExpertScore]] = {p: [] for p in image_paths}
    use_bar = os.environ.get("USE_PROGRESS_BAR", "1").lower() in {"1", "true", "yes"}
    iterator = (
        tqdm(experts, desc="Experts", ncols=80, leave=False)
        if (tqdm and use_bar) else experts
    )
    for ex in iterator:
        prov = get_provider(ex.provider, **(ex.kwargs or {}))
        res = prov.compute_scores(image_paths)
        for p in image_paths:
            sr = res.get(p)
            sc = None if sr is None else sr.score
            out[p].append(ExpertScore(provider=ex.provider, alias=ex.alias, score=sc))
    return out


def resolve_abs_paths(paths: Iterable[str], root_prefix: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        if os.path.isabs(p):
            out.append(os.path.normpath(p))
        else:
            if root_prefix:
                out.append(os.path.normpath(os.path.join(root_prefix, p)))
            else:
                out.append(os.path.abspath(p))
    return out
