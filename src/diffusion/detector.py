import math
from typing import Dict, List, Optional

import torch

try:
    # Local Aligner (single or multi‑model); we will use single model
    from src.diffusion.core import Aligner
except Exception:
    # Fallback for older layout
    from src.aligner.core import Aligner  # type: ignore


class DiffusionDetector:
    """Adapter that wraps Aligner for a single diffusion/forensics sub‑model.

    Exposes a Blending‑like batching API and returns {path: {"score": float}}.
    """

    def __init__(self, *, weights_dir: str, model: str, device: Optional[str] = None,
                 use_amp: bool = True, amp_dtype: str = "fp16",
                 channels_last: bool = True, enable_tf32: bool = True,
                 compile_model: bool = False, compile_mode: str = "safe") -> None:
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_name = str(model)
        # Create a single‑model aligner; read config from {weights_dir}/{model}/config.yaml
        self.aligner = Aligner(
            models_list=[self.model_name],
            weights_dir=weights_dir,
            device=self.device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
            enable_tf32=enable_tf32,
            compile_model=compile_model,
            compile_mode=compile_mode,
        )

    def infer(self, image_paths: List[str], *, batch_size: int = 256, num_workers: int = 4,
              pin_memory: bool = True, persistent_workers: bool = True, prefetch_factor: int = 2) -> Dict[str, Dict[str, float]]:
        """Run batched inference and return {path: {"score": float}}.

        Any images that fail to load will be present with {"error": ...}.
        """
        try:
            results = self.aligner.infer(
                image_paths,
                batch_size=max(1, int(batch_size)),
                num_workers=max(0, int(num_workers)),
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
                prefetch_factor=max(2, int(prefetch_factor)) if int(num_workers) > 0 else 2,
            )
        except RuntimeError as e:
            # Fallback when variable image sizes prevent stacking; use per-image path
            if 'stack expects each tensor to be equal size' in str(e):
                results = self.aligner.predict_paths(image_paths)
            else:
                raise
        out: Dict[str, Dict[str, float]] = {}
        for p in image_paths:
            entry = results.get(p)
            if isinstance(entry, dict):
                if 'error' in entry:
                    out[p] = {'error': entry.get('error')}
                    continue
                # prefer explicit score
                sc = entry.get('score')
                if isinstance(sc, (int, float)):
                    out[p] = {'score': float(sc)}
                    continue
                # else try by model name
                mv = entry.get(self.model_name)
                if isinstance(mv, (int, float)):
                    out[p] = {'score': float(mv)}
                    continue
            # default missing
            out[p] = {'score': None}  # type: ignore[assignment]
        return out
