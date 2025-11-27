"""Self‑contained Aligner package

This folder contains the minimal runtime to load detectors and run inference.
It is designed to be easily copied to other projects without dragging the
whole repository.
"""

from .core import Aligner

__all__ = ["Aligner"]

