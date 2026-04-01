"""Neuroplastic vLLM Plugin — In-Process Weight Modification & Activation Tracing

Monkey-patches vLLM EngineCore with weight modification methods, dispatched
via the UTILITY ZMQ message path (call_utility_async).

Deploy: copy into vLLM container, use neuroplastic_entrypoint.py to start.

Dispatch chain:
    API route → engine_client.engine_core.call_utility_async(method, *args)
    → ZMQ → EngineCore._handle_client_request(UTILITY, ...)
    → getattr(self, method_name)(*args)   [self = EngineCore instance]
    → accesses self.model_executor.driver_worker.model_runner.model

Modify operations:
    Whole-tensor:  scale, add
    Per-head:      scale_slice, add_slice, zero_heads
    Interpolation: lerp, clamp
    2D matrix:     scale_rows, scale_cols
    Exploration:   add_noise
    Statistics:    normalize

Trace system:
    trace_start → installs forward hooks on all layers
    (inference runs normally, hooks capture activation data)
    trace_collect → removes hooks, returns per-layer temporal dynamics
"""

import logging
import time

import torch  # type: ignore[import-not-found]

logger = logging.getLogger("neuroplastic")

# Checkpoint storage (in-memory, keyed by name+tensor)
_checkpoints: dict[str, torch.Tensor] = {}


def _navigate_to_param(model: torch.nn.Module, tensor_path: str):
    """Navigate model hierarchy to find a parameter by dotted path."""
    parts = tensor_path.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    if isinstance(current, (torch.nn.Parameter, torch.Tensor)):
        return current
    return None


def _get_model(self):
    """Get the model from EngineCore's executor."""
    try:
        return self.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    try:
        return self.model_executor.driver_worker.model_runner.get_model()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Operation helpers
# ---------------------------------------------------------------------------

def _validate_slice(params: dict, dim_size: int) -> tuple[int, int]:
    """Parse and validate start/end from params. Returns (start, end)."""
    start = int(params.get("start", 0))
    end = int(params.get("end", dim_size))
    if start < 0 or end > dim_size or start >= end:
        raise ValueError(f"Invalid slice [{start}:{end}] for dim size {dim_size}")
    return start, end


def _validate_indices(params: dict, dim_size: int, device) -> torch.Tensor:
    """Parse and validate indices from params. Returns LongTensor on device."""
    indices = params.get("indices", [])
    if not isinstance(indices, list) or len(indices) == 0:
        raise ValueError("'indices' must be a non-empty list of integers")
    for idx in indices:
        if not isinstance(idx, int) or idx < 0 or idx >= dim_size:
            raise ValueError(f"Index {idx} out of range [0, {dim_size})")
    return torch.tensor(indices, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def _op_scale(data: torch.Tensor, params: dict) -> None:
    """Multiply entire tensor by a scalar."""
    data.mul_(params["value"])


def _op_add(data: torch.Tensor, params: dict) -> None:
    """Add a scalar to every element."""
    data.add_(params["value"])


def _op_scale_slice(data: torch.Tensor, params: dict) -> None:
    """Scale a contiguous slice along dim 0 (specific heads)."""
    start, end = _validate_slice(params, data.shape[0])
    data[start:end].mul_(params["value"])


def _op_add_slice(data: torch.Tensor, params: dict) -> None:
    """Add a scalar to a contiguous slice along dim 0."""
    start, end = _validate_slice(params, data.shape[0])
    data[start:end].add_(params["value"])


def _op_zero_heads(data: torch.Tensor, params: dict) -> None:
    """Zero out specific heads (dim 0 indices). Ablation tool.

    For 1D tensors: zeros selected elements.
    For 2D tensors: zeros selected rows (entire output channel).
    """
    idx = _validate_indices(params, data.shape[0], data.device)
    if data.ndim == 1:
        data[idx] = 0
    else:
        data[idx, ...] = 0


def _op_lerp(data: torch.Tensor, params: dict) -> None:
    """Interpolate between current value and a saved checkpoint.

    alpha=0.0 → no change (keep current)
    alpha=1.0 → fully restore checkpoint
    alpha=0.5 → halfway between current and checkpoint
    """
    checkpoint_name = params.get("checkpoint", "")
    tensor_path = params["_tensor_path"]
    alpha = float(params.get("alpha", 0.5))
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    key = f"{checkpoint_name}_{tensor_path}"
    if key not in _checkpoints:
        raise ValueError(f"Checkpoint '{checkpoint_name}' not found for '{tensor_path}'")
    data.lerp_(_checkpoints[key].to(data.device, data.dtype), alpha)


def _op_clamp(data: torch.Tensor, params: dict) -> None:
    """Clamp values to [min, max] range."""
    lo = params.get("min")
    hi = params.get("max")
    if lo is not None:
        lo = float(lo)
    if hi is not None:
        hi = float(hi)
    if lo is None and hi is None:
        raise ValueError("At least one of 'min' or 'max' must be specified")
    data.clamp_(min=lo, max=hi)


def _op_scale_rows(data: torch.Tensor, params: dict) -> None:
    """Scale specific rows of a 2D weight matrix."""
    if data.ndim < 2:
        raise ValueError(f"scale_rows requires 2D+ tensor, got {data.ndim}D")
    idx = _validate_indices(params, data.shape[0], data.device)
    data[idx] = data[idx] * params["value"]


def _op_scale_cols(data: torch.Tensor, params: dict) -> None:
    """Scale specific columns of a 2D weight matrix."""
    if data.ndim != 2:
        raise ValueError(f"scale_cols requires 2D tensor, got {data.ndim}D")
    idx = _validate_indices(params, data.shape[1], data.device)
    data[:, idx] = data[:, idx] * params["value"]


def _op_add_noise(data: torch.Tensor, params: dict) -> None:
    """Add Gaussian noise scaled relative to the tensor's current std.

    scale=0.01 means noise std is 1% of the tensor's std.
    Useful for stochastic exploration around a promising configuration.
    """
    scale = float(params.get("scale", 0.01))
    if scale <= 0 or scale > 1.0:
        raise ValueError(f"noise scale must be in (0, 1.0], got {scale}")
    seed = params.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
    tensor_std = data.float().std().item()
    noise = torch.randn_like(data, dtype=torch.float32) * tensor_std * scale
    data.add_(noise.to(data.dtype))


def _op_normalize(data: torch.Tensor, params: dict) -> None:
    """Rescale the tensor to a target L2 norm, preserving direction."""
    target = float(params.get("target_norm", 1.0))
    if target <= 0:
        raise ValueError(f"target_norm must be positive, got {target}")
    current_norm = data.float().norm().item()
    if current_norm < 1e-12:
        raise ValueError("Tensor has near-zero norm, cannot normalize")
    data.mul_(target / current_norm)


# Operation dispatch table
_OP_DISPATCH = {
    "scale": _op_scale,
    "add": _op_add,
    "scale_slice": _op_scale_slice,
    "add_slice": _op_add_slice,
    "zero_heads": _op_zero_heads,
    "lerp": _op_lerp,
    "clamp": _op_clamp,
    "scale_rows": _op_scale_rows,
    "scale_cols": _op_scale_cols,
    "add_noise": _op_add_noise,
    "normalize": _op_normalize,
}


# ---------------------------------------------------------------------------
# Methods to be monkey-patched onto EngineCore
# ---------------------------------------------------------------------------

def neuroplastic_modify_weight(self, tensor_path: str, operation: str, params: dict) -> dict:
    """Modify a model weight tensor in-place on GPU.

    Args:
        tensor_path: dotted path to the tensor (e.g. model.layers.50.mixer.A)
        operation: one of the registered operation names
        params: operation-specific parameters dict
    """
    t0 = time.time()
    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    param = _navigate_to_param(model, tensor_path)
    if param is None:
        return {"error": f"Tensor '{tensor_path}' not found"}

    op_fn = _OP_DISPATCH.get(operation)
    if op_fn is None:
        return {"error": f"Unknown operation: {operation}. "
                f"Available: {', '.join(sorted(_OP_DISPATCH))}"}

    # Inject tensor_path for operations that need it (lerp)
    params["_tensor_path"] = tensor_path

    with torch.no_grad():
        before_mean = param.data.float().mean().item()
        before_std = param.data.float().std().item()
        before_norm = param.data.float().norm().item()

        try:
            op_fn(param.data, params)
        except (ValueError, IndexError, RuntimeError) as exc:
            return {"error": f"Operation '{operation}' failed: {exc}"}

        after_mean = param.data.float().mean().item()
        after_std = param.data.float().std().item()
        after_norm = param.data.float().norm().item()

    elapsed = time.time() - t0
    # Clean internal keys from params before returning
    result_params = {k: v for k, v in params.items() if not k.startswith("_")}
    return {
        "status": "ok",
        "tensor": tensor_path,
        "operation": operation,
        "params": result_params,
        "shape": list(param.shape),
        "dtype": str(param.dtype),
        "before": {"mean": before_mean, "std": before_std, "norm": before_norm},
        "after": {"mean": after_mean, "std": after_std, "norm": after_norm},
        "elapsed_ms": elapsed * 1000,
    }


def neuroplastic_inspect_weight(self, tensor_path: str, per_head: bool = False) -> dict:
    """Inspect a model weight tensor.

    Args:
        tensor_path: dotted path to the tensor
        per_head: if True, include per-element values for small 1D tensors
    """
    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    param = _navigate_to_param(model, tensor_path)
    if param is None:
        return {"error": f"Tensor '{tensor_path}' not found"}

    with torch.no_grad():
        fp = param.data.float()
        stats = {
            "tensor": tensor_path,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "device": str(param.device),
            "mean": fp.mean().item(),
            "std": fp.std().item(),
            "min": fp.min().item(),
            "max": fp.max().item(),
            "abs_max": fp.abs().max().item(),
            "norm": fp.norm().item(),
            "numel": param.numel(),
        }
        if len(param.shape) == 2 and param.shape[0] <= 256:
            row_norms = fp.norm(dim=1)
            stats["row_norm_mean"] = row_norms.mean().item()
            stats["row_norm_std"] = row_norms.std().item()
            stats["row_norm_cv_pct"] = (row_norms.std() / row_norms.mean() * 100).item()

        # Per-head values for small 1D tensors
        if per_head and param.ndim == 1 and param.numel() <= 128:
            stats["values"] = [round(v, 6) for v in fp.tolist()]

    return stats


def neuroplastic_save_checkpoint(self, tensor_path: str, name: str) -> dict:
    """Save current tensor values to an in-memory checkpoint."""
    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    param = _navigate_to_param(model, tensor_path)
    if param is None:
        return {"error": f"Tensor '{tensor_path}' not found"}

    key = f"{name}_{tensor_path}"
    _checkpoints[key] = param.data.clone()

    return {
        "status": "saved",
        "tensor": tensor_path,
        "checkpoint": name,
        "shape": list(param.shape),
        "n_checkpoints": len(_checkpoints),
    }


def neuroplastic_restore_checkpoint(self, tensor_path: str, name: str) -> dict:
    """Restore tensor values from an in-memory checkpoint."""
    key = f"{name}_{tensor_path}"
    if key not in _checkpoints:
        return {"error": f"Checkpoint '{name}' not found for '{tensor_path}'"}

    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    param = _navigate_to_param(model, tensor_path)
    if param is None:
        return {"error": f"Tensor '{tensor_path}' not found"}

    with torch.no_grad():
        param.data.copy_(_checkpoints[key])

    return {
        "status": "restored",
        "tensor": tensor_path,
        "checkpoint": name,
        "shape": list(param.shape),
    }


def neuroplastic_list_tensors(self, filter_str: str = "") -> dict:
    """List model tensor names."""
    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    tensors = []
    for name, param in model.named_parameters():
        if filter_str and filter_str not in name:
            continue
        tensors.append({
            "name": name,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
        })

    return {"tensors": tensors, "count": len(tensors)}


def neuroplastic_save_to_disk(self, path: str, tensor_paths: list[str]) -> dict:
    """Save specific tensors to a safetensors file on disk."""
    import os
    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    tensors_to_save = {}
    for tp in tensor_paths:
        param = _navigate_to_param(model, tp)
        if param is None:
            return {"error": f"Tensor '{tp}' not found"}
        tensors_to_save[tp] = param.data.contiguous().cpu()

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    from safetensors.torch import save_file
    save_file(tensors_to_save, path)

    return {
        "status": "saved",
        "path": path,
        "tensors": list(tensors_to_save.keys()),
        "count": len(tensors_to_save),
    }


def neuroplastic_load_from_disk(self, path: str) -> dict:
    """Load tensors from a safetensors delta file and apply to the live model."""
    import os
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    from safetensors.torch import load_file
    saved = load_file(path)

    applied = []
    with torch.no_grad():
        for tp, tensor_data in saved.items():
            param = _navigate_to_param(model, tp)
            if param is None:
                return {"error": f"Tensor '{tp}' not found in model"}
            param.data.copy_(tensor_data.to(param.device, param.dtype))
            applied.append(tp)

    return {
        "status": "loaded",
        "path": path,
        "tensors": applied,
        "count": len(applied),
    }


# ---------------------------------------------------------------------------
# Activation trace system — manual forward pass (bypasses CUDA graphs)
# ---------------------------------------------------------------------------

# Layer type constants
MAMBA_LAYERS = {0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50}
ATTENTION_LAYERS = {5,12,19,26,33,42}
MOE_LAYERS = {1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51}


def _layer_type(idx: int) -> str:
    if idx in MAMBA_LAYERS:
        return "mamba"
    elif idx in ATTENTION_LAYERS:
        return "attention"
    elif idx in MOE_LAYERS:
        return "moe"
    return "unknown"


# Trace state (module-level, persists between calls within EngineCore process)
_trace_hooks: list = []
_trace_data: dict = {}
_trace_prev_residual: dict = {}


def neuroplastic_trace_start(self) -> dict:
    """Install forward hooks on all layers to capture activation trajectories.

    Hooks fire during the next inference pass (requires --enforce-eager).
    Call neuroplastic_trace_collect() after inference to retrieve data.
    """
    global _trace_hooks, _trace_data, _trace_prev_residual
    for h in _trace_hooks:
        h.remove()
    _trace_hooks = []
    _trace_data = {}
    _trace_prev_residual = {}

    model = _get_model(self)
    if model is None:
        return {"error": "Cannot access model object"}

    n_layers = 0
    for i, layer in enumerate(model.model.layers):
        def make_hook(layer_idx):
            def hook_fn(module, inp, output):
                with torch.no_grad():
                    h = output[0] if isinstance(output, tuple) else output
                    residual = output[1] if isinstance(output, tuple) and len(output) > 1 else None

                    # Only capture prefill (longest sequence), skip decode steps
                    if layer_idx in _trace_data:
                        prev_len = len(_trace_data[layer_idx].get("output_norms", []))
                        if h.shape[0] <= prev_len:
                            return

                    fp = h.float()
                    num_tokens = fp.shape[0]
                    hidden_dim = fp.shape[-1]

                    entry = {
                        "type": _layer_type(layer_idx),
                        "output_norms": fp.norm(dim=-1).tolist(),
                        "output_var": fp.var(dim=-1).tolist(),
                    }

                    # State change rate
                    if num_tokens > 1:
                        diffs = (fp[1:] - fp[:-1]).norm(dim=-1)
                        entry["change_rate"] = [0.0] + diffs.tolist()
                    else:
                        entry["change_rate"] = [0.0]

                    # Residual stream
                    if residual is not None:
                        res_fp = residual.float()
                        entry["residual_norms"] = res_fp.norm(dim=-1).tolist()

                        if layer_idx > 0 and (layer_idx - 1) in _trace_prev_residual:
                            prev_res = _trace_prev_residual[layer_idx - 1]
                            if prev_res.shape == res_fp.shape:
                                cos = torch.nn.functional.cosine_similarity(
                                    res_fp, prev_res, dim=-1
                                )
                                entry["residual_cosine_prev"] = cos.tolist()

                        _trace_prev_residual[layer_idx] = res_fp.clone()

                    # Mamba per-head magnitude
                    lt = _layer_type(layer_idx)
                    if lt == "mamba" and hidden_dim >= 64:
                        num_heads = 64
                        head_dim = hidden_dim // num_heads
                        if num_heads * head_dim == hidden_dim:
                            reshaped = fp.view(num_tokens, num_heads, head_dim)
                            head_norms = reshaped.norm(dim=-1)
                            mean_per_head = head_norms.mean(dim=0)
                            sorted_heads = mean_per_head.argsort(descending=True)
                            top5 = sorted_heads[:5].tolist()
                            bot5 = sorted_heads[-5:].tolist()
                            entry["top_heads"] = {
                                "indices": top5,
                                "norms": head_norms[:, sorted_heads[:5]].tolist(),
                            }
                            entry["bottom_heads"] = {
                                "indices": bot5,
                                "norms": head_norms[:, sorted_heads[-5:]].tolist(),
                            }
                            entry["head_norm_mean"] = mean_per_head.tolist()

                    _trace_data[layer_idx] = entry
            return hook_fn

        _trace_hooks.append(layer.register_forward_hook(make_hook(i)))
        n_layers += 1

    return {"status": "hooks_installed", "n_layers": n_layers}


def neuroplastic_trace_collect(self) -> dict:
    """Remove hooks and return the captured trace data."""
    global _trace_hooks, _trace_prev_residual
    for h in _trace_hooks:
        h.remove()
    _trace_hooks = []
    _trace_prev_residual = {}

    if not _trace_data:
        return {"error": "No trace data collected. Ensure --enforce-eager is set."}

    layers = {}
    residual_norms = []

    for layer_idx in sorted(_trace_data.keys()):
        entry = _trace_data[layer_idx]
        layers[f"layer_{layer_idx}"] = entry
        if "residual_norms" in entry:
            norms = entry["residual_norms"]
            residual_norms.append({
                "layer": layer_idx,
                "mean_norm": sum(norms) / len(norms) if norms else 0,
            })

    residual_stream = {}
    if residual_norms:
        residual_stream["norm_per_layer"] = [r["mean_norm"] for r in residual_norms]
        cosine_sims = []
        for layer_idx in sorted(_trace_data.keys()):
            entry = _trace_data[layer_idx]
            if "residual_cosine_prev" in entry:
                cos_vals = entry["residual_cosine_prev"]
                cosine_sims.append({
                    "layer": layer_idx,
                    "mean_cosine": sum(cos_vals) / len(cos_vals) if cos_vals else 0,
                })
        residual_stream["cosine_sim_adjacent"] = cosine_sims

    result = {
        "status": "ok",
        "n_layers_captured": len(layers),
        "layers": layers,
        "residual_stream": residual_stream,
    }

    _trace_data.clear()
    return result


def install_engine_core_methods():
    """Monkey-patch EngineCore with neuroplastic methods."""
    from vllm.v1.engine.core import EngineCore

    EngineCore.neuroplastic_modify_weight = neuroplastic_modify_weight
    EngineCore.neuroplastic_inspect_weight = neuroplastic_inspect_weight
    EngineCore.neuroplastic_save_checkpoint = neuroplastic_save_checkpoint
    EngineCore.neuroplastic_restore_checkpoint = neuroplastic_restore_checkpoint
    EngineCore.neuroplastic_list_tensors = neuroplastic_list_tensors
    EngineCore.neuroplastic_save_to_disk = neuroplastic_save_to_disk
    EngineCore.neuroplastic_load_from_disk = neuroplastic_load_from_disk
    EngineCore.neuroplastic_trace_start = neuroplastic_trace_start
    EngineCore.neuroplastic_trace_collect = neuroplastic_trace_collect

    logger.info("Neuroplastic methods installed on EngineCore "
                f"({len(_OP_DISPATCH)} modify operations, trace enabled)")


def install_api_routes(app):
    """Add neuroplastic HTTP endpoints to the FastAPI app."""
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.post("/neuroplastic/modify")
    async def neuroplastic_modify(request: Request):
        body = await request.json()
        tensor = body.get("tensor", "")
        op = body.get("op", "scale")

        # Build params dict — new style uses "params", old style uses "value"
        params = body.get("params", {})
        if not params and "value" in body:
            params = {"value": float(body["value"])}

        engine_client = request.app.state.engine_client
        await engine_client.pause_generation()
        try:
            result = await engine_client.engine_core.call_utility_async(
                "neuroplastic_modify_weight", tensor, op, params
            )
        finally:
            await engine_client.resume_generation()

        return JSONResponse(content=result)

    @app.post("/neuroplastic/inspect")
    async def neuroplastic_inspect(request: Request):
        body = await request.json()
        tensor = body.get("tensor", "")
        per_head = body.get("per_head", False)
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_inspect_weight", tensor, per_head
        )
        return JSONResponse(content=result)

    @app.post("/neuroplastic/checkpoint")
    async def neuroplastic_checkpoint(request: Request):
        body = await request.json()
        tensor = body.get("tensor", "")
        name = body.get("name", "default")
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_save_checkpoint", tensor, name
        )
        return JSONResponse(content=result)

    @app.post("/neuroplastic/restore")
    async def neuroplastic_restore(request: Request):
        body = await request.json()
        tensor = body.get("tensor", "")
        name = body.get("name", "default")
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_restore_checkpoint", tensor, name
        )
        return JSONResponse(content=result)

    @app.post("/neuroplastic/list")
    async def neuroplastic_list(request: Request):
        body = await request.json()
        filter_str = body.get("filter", "")
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_list_tensors", filter_str
        )
        return JSONResponse(content=result)

    @app.post("/neuroplastic/save")
    async def neuroplastic_save(request: Request):
        body = await request.json()
        path = body.get("path", "")
        tensors = body.get("tensors", [])
        if not path:
            return JSONResponse(content={"error": "path is required"}, status_code=400)
        if not tensors:
            return JSONResponse(content={"error": "tensors list is required"}, status_code=400)

        engine_client = request.app.state.engine_client
        await engine_client.pause_generation()
        try:
            result = await engine_client.engine_core.call_utility_async(
                "neuroplastic_save_to_disk", path, tensors
            )
        finally:
            await engine_client.resume_generation()

        return JSONResponse(content=result)

    @app.post("/neuroplastic/load")
    async def neuroplastic_load(request: Request):
        body = await request.json()
        path = body.get("path", "")
        if not path:
            return JSONResponse(content={"error": "path is required"}, status_code=400)

        engine_client = request.app.state.engine_client
        await engine_client.pause_generation()
        try:
            result = await engine_client.engine_core.call_utility_async(
                "neuroplastic_load_from_disk", path
            )
        finally:
            await engine_client.resume_generation()

        return JSONResponse(content=result)

    @app.post("/neuroplastic/trace/start")
    async def neuroplastic_trace_start_route(request: Request):
        """Install forward hooks for activation tracing.

        Requires --enforce-eager (no CUDA graphs) for hooks to fire.
        """
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_trace_start"
        )
        return JSONResponse(content=result)

    @app.post("/neuroplastic/trace/collect")
    async def neuroplastic_trace_collect_route(request: Request):
        """Remove hooks and return captured trace data."""
        engine_client = request.app.state.engine_client
        result = await engine_client.engine_core.call_utility_async(
            "neuroplastic_trace_collect"
        )
        return JSONResponse(content=result)

    logger.info("Neuroplastic API routes installed (including trace)")
