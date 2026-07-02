"""Locks livianos para operaciones administrativas del MVP."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Union


class PipelineFileLock:
    """Lock intra-proceso + archivo atomico para evitar pipelines paralelos."""

    def __init__(self, path: Union[str, Path], *, stale_seconds: int = 12 * 60 * 60) -> None:
        self.path = Path(path)
        self.stale_seconds = max(60, stale_seconds)
        self._thread_lock = threading.Lock()
        self._has_file_lock = False

    def acquire(self, blocking: bool = True) -> bool:
        if not self._thread_lock.acquire(blocking=blocking):
            return False

        if self._acquire_file_lock():
            return True

        self._thread_lock.release()
        return False

    def release(self) -> None:
        try:
            if self._has_file_lock:
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                finally:
                    self._has_file_lock = False
        finally:
            self._thread_lock.release()

    def _acquire_file_lock(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._is_stale():
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                return self._acquire_file_lock()
            return False

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\ncreated_at={int(time.time())}\n")
        self._has_file_lock = True
        return True

    def _is_stale(self) -> bool:
        try:
            return (time.time() - self.path.stat().st_mtime) > self.stale_seconds
        except FileNotFoundError:
            return False
