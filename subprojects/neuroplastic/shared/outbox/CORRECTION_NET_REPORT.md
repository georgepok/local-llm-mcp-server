# Experiment A: Output Correction Network — Results

## Summary

The correction net produced **zero improvement** over the frozen base model. The output computation bottleneck cannot be fixed by an additive correction head on frozen features.

## Final Results (40 eval batches, 320 tasks)

| Metric | Base Model | Corrected | Delta |
|--------|-----------|-----------|-------|
| Xform accuracy | 47.9% | 47.7% | **-0.2%** |
| Tasks solved | 2/320 | 2/320 | **+0** |

## Training Trajectory

| Epoch | Loss | Base Xform | Corr Xform | Delta |
|-------|------|-----------|------------|-------|
| 0 | 1.572 | 49.9% | 50.0% | +0.1% |
| 10 | 1.544 | 46.6% | 47.0% | +0.4% |
| 30 | 1.420 | 46.1% | 45.8% | -0.3% |
| 50 | 1.375 | 56.0% | 56.2% | +0.2% |
| 70 | 1.380 | 48.8% | 47.8% | -1.0% |
| 90 | 1.351 | 51.8% | 50.7% | -1.1% |
| 99 | 1.348 | 50.5% | 51.6% | +1.1% |

The correction delta oscillates ±1% across all epochs — pure noise. Training loss decreases (1.57→1.35) but doesn't translate to eval improvement. The correction net overfits to training distribution.

## Configuration

- Base checkpoint: `output_30to50/checkpoints/best.pt` (step 15000, 30→50% sequential)
- Correction net: 202,506 params (pred_embed + 3 linear layers)
- Architecture: `concat(h_final.detach(), embed(base_pred)) → FC(512→256) → GELU → FC(256→256) → GELU → FC(256→10)`
- Output layer zero-initialized (additive identity at start)
- Training: 100 epochs × 50 batches × batch_size=16, LR=1e-3, transform weight=5.0
- Total training time: ~47 minutes

## Interpretation

The base model's 256-dim hidden states at the ODE output **do not contain extractable information about the correct transformation rule** beyond what the base model's own output head already uses. A 202K-param correction network with access to both the hidden state and the base prediction cannot improve accuracy.

This rules out Level 2 (refinement head) approaches for this architecture scale. The bottleneck is not "almost right answers that need refinement" — it's "wrong transformation rules encoded in insufficient capacity."

The verified TTT finding (77% accuracy on converged tasks) shows the model CAN achieve higher accuracy when its geometry is task-specialized via gradient descent. But this task-specific information lives in the adapted MetricNet/TauNet/W_o weights, not in the frozen hidden state features that the correction net sees.
