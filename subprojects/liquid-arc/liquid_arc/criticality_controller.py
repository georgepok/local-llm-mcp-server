"""Adaptive criticality controller for LiquidARC training.

Maintains the training system in the critical CV zone (default 4.5–6.0) by
dynamically adjusting the real ARC mixing ratio. Uses model-based feedforward
control with smooth EMA transitions and asymmetric correction rates.

Empirical basis (from Phase 1a / 1a.1 / 1a.2 experiments):
  - CV < 3.5: sub-critical — flat metric, weak geometric signal
  - CV 4.5–6.0: critical zone — structured geometry, strong correlations
  - CV > 6.5: crystallizing / diverging — metric overfit risk

Design principles:
  - Model-based: target ratio derived from known system dynamics, not PID error
  - Smooth: alpha=0.02 default, max ~2% change per update (prevents oscillation)
  - Anticipatory: incorporates CV trend (rate of change), not just current value
  - Asymmetric: corrects faster when CV drops below zone (fast/dangerous failure)
  - Noise-filtered: smooths CV over configurable history window before acting
  - CSV logging: optional incremental write to controller_log.csv for analysis

Usage:
    controller = CriticalityController(
        initial_ratio=config.real_arc_mix_ratio,
        log_path=os.path.join(out_dir, "controller_log.csv"),
    )
    # Inside training loop:
    ratio = controller.update(cv_val, xform_acc_val, step)
    use_real = random.random() < ratio
    # For logging:
    diag = controller.diagnostics()
    writer.add_scalar("ctrl/ratio", diag["arc_ratio"], step)
    # At end of training:
    controller.close()
"""

from __future__ import annotations

import csv
import os
from collections import deque
from typing import Dict, Optional

import numpy as np


class CriticalityController:
    """Model-based adaptive controller for real ARC mix ratio.

    Args:
        cv_zone_low: Lower bound of critical CV zone.
        cv_zone_high: Upper bound of critical CV zone.
        cv_target: Center of critical zone (for reference only).
        ratio_min: Hard floor on mix ratio.
        ratio_max: Hard ceiling on mix ratio.
        initial_ratio: Starting mix ratio.
        smoothing_window: Steps of CV history for smoothed estimate.
        trend_window: Steps of CV history for rate-of-change estimate.
        alpha: Base EMA factor toward target (controls smoothness).
        update_every: Recalculate ratio only every N steps.
        log_path: If provided, write controller trajectory to this CSV path
                  incrementally (one row per update_every interval).
    """

    def __init__(
        self,
        cv_zone_low: float = 4.5,
        cv_zone_high: float = 6.0,
        cv_target: float = 5.25,
        ratio_min: float = 0.15,
        ratio_max: float = 0.65,
        initial_ratio: float = 0.30,
        smoothing_window: int = 50,
        trend_window: int = 100,
        alpha: float = 0.02,
        update_every: int = 10,
        log_path: Optional[str] = None,
    ) -> None:
        self.cv_zone_low = cv_zone_low
        self.cv_zone_high = cv_zone_high
        self.cv_target = cv_target
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.alpha = alpha
        self.update_every = update_every

        self.arc_ratio: float = initial_ratio

        # Rolling history deques
        self.cv_history: deque = deque(maxlen=max(smoothing_window, trend_window))
        self.ratio_history: deque = deque(maxlen=500)
        self.xform_history: deque = deque(maxlen=500)

        self._smoothing_window = smoothing_window
        self._trend_window = trend_window
        self.step_count: int = 0

        # In-memory log entries (also written to CSV if log_path set)
        self.log_entries: list = []

        # CSV incremental writer
        self._csv_file = None
        self._csv_writer = None
        if log_path is not None:
            self._open_csv(log_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_cv: float,
        xform_acc: Optional[float] = None,
        step: Optional[int] = None,
    ) -> float:
        """Update controller with new CV observation and return mix ratio.

        Call every training step. The ratio is only recomputed every
        ``update_every`` steps to reduce noise sensitivity.

        Args:
            raw_cv: Current metric CV (from result["metric_cv"]).
            xform_acc: Current transform accuracy (optional, for logging).
            step: Global training step (optional, for logging).

        Returns:
            Float in [ratio_min, ratio_max]: probability of sampling real ARC.
        """
        self.cv_history.append(raw_cv)
        if xform_acc is not None:
            self.xform_history.append(xform_acc)
        self.step_count += 1

        # Only recalculate ratio on schedule
        if self.step_count % self.update_every != 0:
            return self.arc_ratio

        # Need minimum history for reliable estimates
        if len(self.cv_history) < 30:
            return self.arc_ratio

        smooth_cv = self._smooth_cv()
        cv_rate = self._cv_rate()
        target_ratio = self._get_target_ratio(smooth_cv, cv_rate)

        # Asymmetric alpha: faster correction when CV is below zone
        # (sub-critical flat-metric is the primary failure mode from Phase 1a.2)
        if smooth_cv < self.cv_zone_low:
            effective_alpha = self.alpha * 2.0
        elif smooth_cv > self.cv_zone_high:
            effective_alpha = self.alpha * 1.5
        else:
            effective_alpha = self.alpha

        # EMA toward target
        self.arc_ratio = (
            self.arc_ratio * (1.0 - effective_alpha)
            + target_ratio * effective_alpha
        )
        self.arc_ratio = float(np.clip(self.arc_ratio, self.ratio_min, self.ratio_max))

        self.ratio_history.append(self.arc_ratio)

        # Build log entry
        zone = self._get_zone(smooth_cv)
        entry = {
            "step": step if step is not None else self.step_count,
            "raw_cv": raw_cv,
            "smooth_cv": smooth_cv,
            "cv_rate": cv_rate,
            "arc_ratio": self.arc_ratio,
            "target_ratio": target_ratio,
            "zone": zone,
            "xform_acc": xform_acc if xform_acc is not None else 0.0,
        }
        self.log_entries.append(entry)

        # Write to CSV incrementally
        if self._csv_writer is not None:
            self._csv_writer.writerow(entry)
            self._csv_file.flush()

        return self.arc_ratio

    def diagnostics(self) -> Dict:
        """Return controller state dict for logging and tensorboard.

        Keys:
            arc_ratio: current mix ratio
            smooth_cv: EMA-smoothed CV estimate over history window
            cv_rate: estimated CV rate of change (per 1K steps)
            zone: string zone label (sub_critical / critical / crystallizing / warmup)
        """
        if len(self.cv_history) < 30:
            return {
                "arc_ratio": self.arc_ratio,
                "smooth_cv": 0.0,
                "cv_rate": 0.0,
                "zone": "warmup",
            }

        smooth_cv = self._smooth_cv()
        cv_rate = self._cv_rate()

        return {
            "arc_ratio": self.arc_ratio,
            "smooth_cv": smooth_cv,
            "cv_rate": cv_rate,
            "zone": self._get_zone(smooth_cv),
        }

    def save_log(self, path: str) -> None:
        """Write all accumulated log entries to a CSV file.

        This is a convenience method for saving the full trajectory at the end
        of training. If log_path was passed at construction, entries are also
        written incrementally during training.
        """
        if not self.log_entries:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.log_entries[0].keys())
            writer.writeheader()
            writer.writerows(self.log_entries)

    def get_zone_stats(self, since_step: int = 0) -> Dict:
        """Return time-in-zone statistics for post-training reporting.

        Args:
            since_step: Only count log entries at or after this step.

        Returns:
            Dict with total_updates, per-zone percentages, avg_ratio, ratio_range.
        """
        entries = [e for e in self.log_entries if e["step"] >= since_step]
        if not entries:
            return {}

        total = len(entries)
        zones: Dict[str, int] = {"sub_critical": 0, "critical": 0, "crystallizing": 0}
        for e in entries:
            zone = e.get("zone", "unknown")
            if zone in zones:
                zones[zone] += 1

        ratios = [e["arc_ratio"] for e in entries]
        return {
            "total_updates": total,
            "sub_critical_pct": zones["sub_critical"] / total * 100,
            "critical_pct": zones["critical"] / total * 100,
            "crystallizing_pct": zones["crystallizing"] / total * 100,
            "avg_ratio": float(np.mean(ratios)),
            "ratio_range": (min(ratios), max(ratios)),
        }

    def close(self) -> None:
        """Flush and close the incremental CSV log file if open."""
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth_cv(self) -> float:
        """Return mean CV over the smoothing window."""
        window = list(self.cv_history)[-self._smoothing_window:]
        return float(np.mean(window))

    def _cv_rate(self) -> float:
        """Estimate CV rate of change in units of CV per 1K steps.

        Compares mean of older quarter of trend window vs. most recent quarter.
        Returns 0.0 if insufficient history.
        """
        if len(self.cv_history) < self._trend_window:
            return 0.0
        buf = list(self.cv_history)[-self._trend_window:]
        old_window = buf[-self._trend_window:-int(self._trend_window * 0.75)]
        new_window = buf[-int(self._trend_window * 0.25):]
        # Approximate midpoint separation in steps
        steps_between = int(self._trend_window * 0.5)
        return float(
            (np.mean(new_window) - np.mean(old_window))
            / max(steps_between / 1000.0, 1e-6)
        )

    def _get_target_ratio(self, smooth_cv: float, cv_rate: float) -> float:
        """Model-based feedforward: optimal ratio given current CV state and trend.

        Based on empirical data from Phase 1a.2 ablations:
          - Sub-critical needs coherent procedural signal → lower real ARC ratio
          - Critical zone needs moderate diversity → moderate ratio
          - Above critical needs diverse signal to prevent crystallization

        The cv_rate term provides anticipatory correction: if CV is trending
        toward the danger zone we adjust early.
        """
        if smooth_cv < 3.5:
            # Well below critical: build coherently with procedural tasks
            base = 0.20
        elif smooth_cv < 4.5:
            # Approaching critical zone: gradual increase
            t = (smooth_cv - 3.5) / 1.0
            base = 0.20 + 0.15 * t   # 0.20 → 0.35
        elif smooth_cv < 5.25:
            # Lower critical zone: moderate diversity
            base = 0.35
            # If CV is dropping, reduce diversity to support it
            if cv_rate < -0.1:
                base += 0.03 * cv_rate  # cv_rate negative → lower ratio
        elif smooth_cv < 6.0:
            # Upper critical zone: slightly more diversity
            base = 0.40
            # If CV rising toward crystallization, preemptively increase
            if cv_rate > 0.1:
                base += 0.05 * min(cv_rate, 1.0)
        elif smooth_cv < 7.0:
            # Above critical: crystallizing, increase diversity
            t = (smooth_cv - 6.0) / 1.0
            base = 0.45 + 0.15 * t   # 0.45 → 0.60
        else:
            # Well above critical: strong diversity pressure
            base = 0.60

        return float(np.clip(base, self.ratio_min, self.ratio_max))

    def _get_zone(self, smooth_cv: float) -> str:
        """Classify CV into named zone for logging."""
        if smooth_cv < self.cv_zone_low:
            return "sub_critical"
        elif smooth_cv <= self.cv_zone_high:
            return "critical"
        else:
            return "crystallizing"

    def _open_csv(self, path: str) -> None:
        """Open CSV file for incremental writing and write header."""
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._csv_file = open(path, "w", newline="", buffering=1)
        fieldnames = [
            "step", "raw_cv", "smooth_cv", "cv_rate",
            "arc_ratio", "target_ratio", "zone", "xform_acc",
        ]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def __del__(self) -> None:
        self.close()
