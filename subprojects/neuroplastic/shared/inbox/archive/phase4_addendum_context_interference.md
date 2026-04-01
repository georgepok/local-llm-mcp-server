# Phase 4 Minor Addendum: Context Interference Test

**From:** Claude Desktop  
**Date:** 2026-03-11  
**Re:** Additional measurement based on Session 3 finding

---

Session 3 revealed that state_001 passes 10/10 in isolation (PROBE) but fails intermittently inside the full eval harness where it runs after other tests. The model CAN compute the answer — context pollution from prior questions interferes.

Add to Phase 4 experiment protocol:

**Phase E: Context Interference Test**

After the Hebbian adaptation (Phase B), measure state_001 accuracy in TWO conditions:

1. **Isolated:** Ask the bag inventory question with no prior context (clean conversation). Run 10 trials. This is the PROBE condition.

2. **Post-context:** Run 3 sequential reasoning tests, then 3 code generation tests, THEN the bag inventory question. Run 10 trials. This mimics the full eval harness.

Compare the two conditions before and after Hebbian adaptation:
- If baseline shows isolated > post-context (confirming interference), AND
- If Hebbian adaptation closes the gap (post-context accuracy approaches isolated accuracy)
- Then the Hebbian updates improved state dynamics robustness, not just raw computation

This is a stronger test than the full eval because it isolates the specific failure mechanism: residual Mamba state from prior tasks polluting subsequent state-tracking computation.

---

*This adds to, doesn't replace, the existing Phase 4 protocol.*
