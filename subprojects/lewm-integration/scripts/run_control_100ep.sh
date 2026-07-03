#!/bin/bash
# Run 100-episode control eval on all 5 checkpoints.
# Invoked inside fgn-train container via nohup + docker exec -d.

set -u
cd /workspace/lewm-integration/le-wm
export SDL_VIDEODRIVER=dummy
export STABLEWM_HOME=/workspace/models/stable-wm
export PYTHONPATH=/workspace/liquid-arc:/workspace/lewm-integration/scripts:/workspace/lewm-integration/le-wm:/workspace/lewm-integration

LOG=/workspace/lewm-integration/runs/control_100ep.log
echo "=== 100-EPISODE CONTROL COMPARISON $(date) ===" > "$LOG"
for p in liquid_long liquid_crit liquid_20k ar_matched ar_20k; do
  echo "=== ${p} $(date +%H:%M) ===" >> "$LOG"
  python eval.py policy="${p}" eval.num_eval=100 seed=42 >> "$LOG" 2>&1
  echo "--- ${p} DONE $(date +%H:%M) ---" >> "$LOG"
done
echo "=== ALL DONE $(date) ===" >> "$LOG"
