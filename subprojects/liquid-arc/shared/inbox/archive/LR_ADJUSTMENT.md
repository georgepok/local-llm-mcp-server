# TASK: Add LR Adjustment MCP Tool

**From:** Claude Desktop
**To:** Claude Code
**Date:** 2026-04-02
**Priority:** URGENT — small change, big impact on embedding learning

---

## What

Add one MCP tool to `mcp_serve.py` that adjusts the optimizer's learning rates at runtime. The Mind's optimizer already has separate parameter groups:

```python
# Group 0: embedding.tokenizer (token_embed + token_pos_embed)
# Group 1: embedding (metadata_proj, event_proj, type_embed, pos_embed, norm)
# Group 2: forcing
# Group 3: readout
```

## New Tool: `set_learning_rates`

```python
@mcp.tool()
def set_learning_rates(
    embed_lr: Optional[float] = None,
    other_lr: Optional[float] = None,
) -> str:
    """Adjust optimizer learning rates at runtime.
    
    The embedding table (group 0) benefits from higher LR because it was
    randomly initialized and needs to organize into semantic neighborhoods.
    Other modules (readout, forcing, metadata) should learn more slowly.
    
    The geometry distillation finding: 100× LR ratio for new parameters
    vs transferred parameters. The embedding table is the newest parameter.
    
    Args:
        embed_lr: Learning rate for token embedding table (group 0).
                  Recommended range: 1e-4 to 1e-2.
        other_lr: Learning rate for readout, forcing, other embedding modules.
                  Recommended range: 1e-6 to 1e-4.
    
    Returns current LR for all groups.
    """
    if _mind.optimizer is None:
        return json.dumps({'status': 'no_optimizer', 'message': 'Online learning disabled'})
    
    if embed_lr is not None:
        _mind.optimizer.param_groups[0]['lr'] = embed_lr
    
    if other_lr is not None:
        for i in range(1, len(_mind.optimizer.param_groups)):
            _mind.optimizer.param_groups[i]['lr'] = other_lr
    
    # Report current state
    groups = []
    for i, pg in enumerate(_mind.optimizer.param_groups):
        groups.append({
            'group': i,
            'lr': pg['lr'],
            'n_params': sum(p.numel() for p in pg['params']),
        })
    
    return json.dumps({
        'status': 'updated',
        'groups': groups,
        'ratio': groups[0]['lr'] / groups[1]['lr'] if len(groups) > 1 else None,
    }, indent=2)
```

## Files to Modify

| File | Change |
|------|--------|
| `liquid_arc/mcp_serve.py` | Add `set_learning_rates` tool |

That's it. One tool, ~30 lines. The optimizer parameter groups already exist.
