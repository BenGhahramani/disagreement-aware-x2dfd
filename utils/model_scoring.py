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


ProviderFactory = Callable[..., ScoreProvider]


_REGISTRY: Dict[str, ProviderFactory] = {
    'blending': lambda **kw: BlendingProvider(
        model_name=kw.get('model_name', 'swinv2_base_window16_256'),
        weights_path=kw.get('weights_path', 'weights/blending_models/best_gf.pth'),
        img_size=int(kw.get('img_size', 256)),
        num_class=int(kw.get('num_class', 2)),
        device=kw.get('device'),
        batch_size=int(kw.get('batch_size', 64)),
        num_workers=int(kw.get('num_workers', 4)),
        pin_memory=bool(kw.get('pin_memory', True)),
    ),
}


def get_provider(name: str, **kwargs) -> ScoreProvider:
    name_l = (name or '').strip().lower()
    if name_l not in _REGISTRY:
        raise KeyError(f"Unknown score provider: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name_l](**kwargs)


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
