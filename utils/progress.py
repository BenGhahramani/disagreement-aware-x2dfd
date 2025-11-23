"""Progress tracking utilities shared across tools (moved from qa_utils)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict


class ProgressTracker:
    """
    Tracks and writes progress to a JSON file periodically.

    Usage:
      tracker = ProgressTracker(Path("results/run_progress.json"))
      tracker.start()
      tracker.update("fake_1", {"current": 1, "total": 10, "desc": "Fake images [1/4]"})
      ...
      tracker.stop()
    """

    def __init__(self, progress_path: Path, interval_seconds: float = 10.0):
        self.progress_path = progress_path
        self.interval_seconds = interval_seconds
        self._lock = threading.Lock()
        self._progress: Dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._last_write = 0.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self.write_progress(force=True)

    def update(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._progress[key] = value

    def write_progress(self, force: bool = False) -> None:
        with self._lock:
            now = time.time()
            if force or now - self._last_write >= self.interval_seconds:
                try:
                    self.progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.progress_path, "w", encoding="utf-8") as f:
                        json.dump(self._progress, f, ensure_ascii=False, indent=2)
                except Exception:
                    # Swallow write errors to avoid interrupting long runs
                    pass
                self._last_write = now

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.write_progress()
            time.sleep(1)
