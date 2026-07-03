"""Lightning callback that prints per-step losses to stdout.

Lightning's default progress bar uses carriage returns, so when stdout is
redirected to a file you never see step-level numbers — only per-epoch
summaries. This callback emits one line every N steps with all metrics
that contain 'loss' in their key, flushed immediately.

Also writes a CSV `metrics.csv` under the trainer's save_dir so curves can
be plotted after the fact.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Optional

import lightning as pl


class LiveStepLogger(pl.pytorch.callbacks.Callback):
    def __init__(self, every_n_steps: int = 50, csv_path: Optional[str] = None):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.csv_path = csv_path
        self._fh = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: list[str] = []
        self._t0: float = time.time()
        self._last_step: int = 0
        self._last_t: float = time.time()

    def _open_csv(self, fields: list[str]):
        if self.csv_path is None or self._fh is not None:
            return
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        is_new = not os.path.exists(self.csv_path)
        self._fh = open(self.csv_path, "a", buffering=1)
        self._fieldnames = ["step", "wall", "epoch"] + fields
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        if is_new:
            self._writer.writeheader()

    def on_train_batch_end(self, trainer: pl.Trainer, module: Any,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        metrics = {k: (v.item() if hasattr(v, "item") else float(v))
                   for k, v in trainer.callback_metrics.items()
                   if "loss" in k and v is not None}
        if not metrics:
            return
        now = time.time()
        dt = max(now - self._last_t, 1e-6)
        dstep = max(step - self._last_step, 1)
        rate = dstep / dt
        self._last_step, self._last_t = step, now

        summary = "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"[step {step:>6} | epoch {trainer.current_epoch} | "
              f"{rate:5.2f} it/s | t+{now - self._t0:7.1f}s]  {summary}",
              flush=True)
        sys.stdout.flush()

        if self.csv_path is not None:
            self._open_csv(sorted(metrics.keys()))
            row = {"step": step, "wall": f"{now - self._t0:.2f}",
                   "epoch": trainer.current_epoch, **metrics}
            assert self._writer is not None
            self._writer.writerow(row)

    def on_train_end(self, trainer: pl.Trainer, module: Any) -> None:
        if self._fh is not None:
            self._fh.close()
