# Self-Organizing Phase Transitions in Neural Computational Geometry — DRAFT STATUS

## Paper Location
The current draft is at: `/mnt/user-data/outputs/phase_transition_paper.md`

## What It Covers
- Full mechanistic characterization of the phase transition phenomenon
- Order parameter (CV), critical threshold, causal mechanism
- Complete landscape across 15/30/50/70% task diversity
- Sequential curriculum results
- Three necessary conditions framework
- Connections to SOC, critical brain hypothesis, grokking

## What It Needs Before Submission

### Essential Experiments
1. **Threshold scaling law**: Need d=128 and d=512 data points to establish CV_critical(d) relationship
2. **Non-ARC task validation**: Same architecture on a different spatial task to prove generality
3. **Minimal architecture ablation**: Which components are necessary? (ODE steps, tau, curvature penalty)
4. **Order parameter prediction test**: Predict transition step from CV trajectory BEFORE it happens

### Writing Needs
- Figures (training curves, CV trajectory, landscape comparison, mechanism diagram)
- Formal mathematical notation for the mechanism section
- Related work section (grokking, emergence, SOC, critical brain literature)
- Proper citations throughout

### Open Questions
- Is the phenomenon specific to SDPA heat kernel routing, or general to any variable-topology architecture?
- Can sustained criticality be achieved (maintain the 16 pp/1K learning rate)?
- What is the theoretical relationship between CV_critical and dimension d?
