"""Layer-Wise ODE Co-Processing — LiquidARC as parallel geometric processor.

Pure perturbation architecture: the ODE accumulates a geometric CORRECTION
to the LLM's residual stream, not an independent trajectory.

    h_ode = h_residual + epsilon * correction

Where correction accumulates additively through depth:
    Layer 1:  correction += dh_1 * dt
    Layer 2:  correction += dh_2 * dt  (carries layer 1's geometric info)
    ...
    Layer 36: correction = sum of all geometric corrections through depth

D^2 is anchored by the residual stream (which dominates at small epsilon).
The MetricNet's g-weighted distances operate on residual-scale vectors.
Even with MetricNet amplification, the correction terms stay small
relative to the residual base.

Starting point: Qwen3-4B (36 layers, d=2560, pure attention).
ODE trained at d=2048 -> projection Linear(2560, 2048).
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .dynamics import ContinuousDynamics
from .context_pool import ContextPool


class LayerWiseODE:
    """LiquidARC ODE as perturbation engine alongside the LLM.

    One Euler step per LLM layer. The ODE correction accumulates
    through depth but h_ode = h_residual + eps*correction keeps
    D^2 anchored by the residual stream.
    """

    def __init__(self, dynamics: ContinuousDynamics, context_pool: ContextPool,
                 n_layers: int, d_llm: int, d_ode: int,
                 epsilon: float = 0.1, device: str = 'cuda'):
        self.dynamics = dynamics
        self.context_pool = context_pool
        self.n_layers = n_layers
        self.d_llm = d_llm
        self.d_ode = d_ode
        self.epsilon = epsilon
        self.device = device

        # Projection when LLM and ODE dimensions differ.
        # AVOID THIS — projection destroys MetricNet's learned geometry.
        # Train ODE at d_llm instead.
        self.proj_in: Optional[torch.nn.Linear] = None
        if d_llm != d_ode:
            print(f"  WARNING: d_llm={d_llm} != d_ode={d_ode}, using projection (degrades geometry)")
            self.proj_in = torch.nn.Linear(d_llm, d_ode, bias=False).to(device)
            torch.nn.init.xavier_uniform_(self.proj_in.weight)
            param_dtype = next(dynamics.parameters()).dtype
            self.proj_in = self.proj_in.to(param_dtype)

        # Perturbation state
        self.correction: Optional[torch.Tensor] = None  # [B, N, d_ode]
        self._prev_h: Optional[torch.Tensor] = None     # previous layer's residual (for delta)
        self.h_state: Optional[torch.Tensor] = None      # persistent [1, K, d_ode]
        self.layer_biases: List[torch.Tensor] = []
        self._layer_diagnostics: List[Dict] = []
        self._active = False
        self.training_mode = False  # True = differentiable path for MetricNet training

    def start_forward(self, h_state: Optional[torch.Tensor] = None):
        """Called at the start of each LLM forward pass."""
        self.correction = None
        self.layer_biases = []
        self._layer_diagnostics = []
        self._active = True
        if h_state is not None:
            self.h_state = h_state

    def process_layer(self, layer_idx: int, h_residual: torch.Tensor) -> torch.Tensor:
        """Called at each LLM layer. Returns attention bias.

        Pure perturbation: h_ode = h_residual + eps * correction.
        Correction accumulates through depth. D^2 anchored by residual.
        """
        B, N, _ = h_residual.shape

        h_res = h_residual.detach() if self.training_mode else h_residual
        param_dtype = next(self.dynamics.parameters()).dtype
        h_res = h_res.to(param_dtype)
        h_proj = self.proj_in(h_res) if self.proj_in is not None else h_res

        if layer_idx == 0:
            self.correction = None
            if len(self.layer_biases) > self.n_layers:
                self.layer_biases = self.layer_biases[-self.n_layers:]
                self._layer_diagnostics = self._layer_diagnostics[-self.n_layers:]

        if self.correction is None:
            self.correction = torch.zeros_like(h_proj)
            if not self.training_mode:
                with torch.no_grad():
                    mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                    context = self.context_pool(h_proj, mask)
                    self.dynamics.set_context(context, mask=None)
            else:
                mask = torch.ones(B, N, dtype=torch.bool, device=self.device)
                context = self.context_pool(h_proj, mask)
                self.dynamics.set_context(context, mask=None)
            self.dynamics.set_n_steps(self.n_layers)

        h_ode = h_proj + self.epsilon * self.correction

        self.dynamics.set_step_index(layer_idx, self.n_layers)
        if hasattr(self.dynamics, 'set_step_embed'):
            self.dynamics.set_step_embed(layer_idx, self.n_layers)

        if self.training_mode:
            dh = self.dynamics(float(layer_idx), h_ode)
            dt = 1.0 / self.n_layers
            self.correction = self.correction + dh * dt
            h_corrected = h_proj + self.epsilon * self.correction
            bias, diag = self._compute_bias(h_corrected)
        else:
            with torch.no_grad():
                dh = self.dynamics(float(layer_idx), h_ode)
                dt = 1.0 / self.n_layers
                self.correction = self.correction + dh * dt
                h_corrected = h_proj + self.epsilon * self.correction
                bias, diag = self._compute_bias(h_corrected)

        with torch.no_grad():
            corr_norm = self.correction.detach().norm(dim=-1).mean().item()
            res_norm = h_proj.detach().norm(dim=-1).mean().item()
            diag['correction_norm'] = corr_norm
            diag['residual_norm'] = res_norm
            diag['correction_ratio'] = corr_norm / (res_norm + 1e-8)
            self.layer_biases.append(bias[0].detach().cpu())
            diag['layer_idx'] = layer_idx
            self._layer_diagnostics.append(diag)

        return bias.unsqueeze(1)

    def end_forward(self) -> Optional[torch.Tensor]:
        """Called after the last layer."""
        self._active = False
        return self.h_state

    def _compute_bias(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute geometric attention bias from ODE state.

        Uses SDPA factorization: B_ij = q_i*k_j/(2t) - ||k_j||^2/(4t)
        where q = k = h_normed * sqrt(g). Same as training.
        """
        B, N, d = h.shape
        param_dtype = next(self.dynamics.parameters()).dtype
        h = h.to(param_dtype)

        # MetricNet forward
        h_normed = self.dynamics.norm_geo(h)
        context = self.dynamics._context
        if context is None:
            context = torch.zeros(B, d, device=h.device, dtype=param_dtype)
        ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
        metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(self.dynamics.metric_net_linear1(metric_input))
        g = F.softplus(self.dynamics.metric_net_linear2_diag(hidden))

        # SDPA factorization
        sqrt_g = g.sqrt()
        qk = h_normed * sqrt_g
        t_diff = F.softplus(self.dynamics.t_diffusion)

        dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
        k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
        bias = dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff)

        # Per-row normalization for softmax compatibility
        N_eff = max(N, 2)
        target_range = 2.0 * math.log(N_eff)
        row_mean = bias.mean(dim=-1, keepdim=True)
        row_centered = bias - row_mean
        row_range = (row_centered.max(dim=-1, keepdim=True).values
                     - row_centered.min(dim=-1, keepdim=True).values).clamp(min=1e-8)
        bias = row_centered / row_range * target_range

        # Diagnostics
        cv = (g.std() / (g.mean() + 1e-8)).item()
        tau = self.dynamics.compute_tau(h)
        tau_mean = tau.mean().item()

        return bias, {
            'cv': cv,
            'tau_mean': tau_mean,
            'B_range': (bias[0].max() - bias[0].min()).item(),
            'B_std': bias[0].std().item(),
        }

    def get_layer_diagnostics(self) -> List[Dict]:
        return self._layer_diagnostics

    def get_summary(self) -> Dict:
        if not self._layer_diagnostics:
            return {}
        cvs = [d['cv'] for d in self._layer_diagnostics]
        taus = [d['tau_mean'] for d in self._layer_diagnostics]
        b_ranges = [d['B_range'] for d in self._layer_diagnostics]
        corr_ratios = [d.get('correction_ratio', 0) for d in self._layer_diagnostics]
        n = len(cvs)
        third = max(n // 3, 1)
        return {
            'n_layers': n,
            'cv_early': sum(cvs[:third]) / third,
            'cv_mid': sum(cvs[third:2*third]) / third,
            'cv_late': sum(cvs[2*third:]) / max(n - 2*third, 1),
            'tau_mean': sum(taus) / n,
            'B_range_early': sum(b_ranges[:third]) / third,
            'B_range_late': sum(b_ranges[2*third:]) / max(n - 2*third, 1),
            'correction_ratio_final': corr_ratios[-1] if corr_ratios else 0,
        }


def hook_llm_layers(llm_model, layer_ode: LayerWiseODE,
                    mode: str = 'residual') -> List:
    """Register hooks on ALL LLM layers for co-processing.

    Two injection modes:
      'residual': POST-hook — add ODE correction directly to hidden states
                  after each layer. Bypasses softmax bottleneck.
      'attention': PRE-hook — add bias to attention_mask.
                   Changes attention weights but fights QK^T scores.

    Residual injection is stronger: the correction enters the residual stream
    directly, so every subsequent layer sees the modified representation.
    """
    hooks = []
    model_layers = llm_model.model.layers

    if mode == 'residual':
        # Post-hook: modify hidden states AFTER the layer.
        # Only INJECT at selected layers (every 6th) to avoid compounding.
        # All layers still RUN the ODE step (for correction accumulation),
        # but only injection layers modify the residual stream.
        n = len(model_layers)
        inject_every = max(1, n // 6)  # ~6 injection points
        inject_set = set(range(inject_every - 1, n, inject_every))

        for i in range(n):
            def make_hook(layer_idx, do_inject):
                def hook_fn(module, input, output):
                    if not layer_ode._active:
                        return output

                    if isinstance(output, tuple):
                        h_out = output[0]
                    else:
                        h_out = output

                    # Run ODE step (always — accumulates correction)
                    _ = layer_ode.process_layer(layer_idx, h_out)

                    # Only inject at selected layers
                    if do_inject and layer_ode.correction is not None:
                        if layer_ode.d_llm != layer_ode.d_ode:
                            return output
                        correction = layer_ode.correction
                        # Normalize correction to a fraction of hidden state norm
                        # This prevents compounding — each injection is bounded
                        h_norm = h_out.detach().norm(dim=-1, keepdim=True).mean()
                        c_norm = correction.detach().norm(dim=-1, keepdim=True).mean()
                        scale = layer_ode.epsilon * h_norm / (c_norm + 1e-8)
                        scale = scale.clamp(max=0.1)  # cap at 10% of hidden norm
                        h_modified = h_out + scale * correction.to(h_out.dtype)

                        if isinstance(output, tuple):
                            return (h_modified,) + output[1:]
                        return h_modified

                    return output

                return hook_fn

            h = model_layers[i].register_forward_hook(
                make_hook(i, i in inject_set))
            hooks.append(h)

    else:  # mode == 'attention'
        # Pre-hook: modify attention_mask (original approach)
        for i in range(len(model_layers)):
            def make_hook(layer_idx):
                def hook_fn(module, args, kwargs):
                    if not layer_ode._active:
                        return

                    hidden_states = args[0] if args else kwargs.get('hidden_states')
                    if hidden_states is None:
                        return

                    bias = layer_ode.process_layer(layer_idx, hidden_states)

                    attn_mask = kwargs.get('attention_mask', None)
                    if attn_mask is not None:
                        _, _, N_bias, _ = bias.shape
                        seq_len = attn_mask.shape[-1]
                        n = min(N_bias, seq_len)
                        if n > 0:
                            injection = torch.zeros_like(attn_mask)
                            injection[:, :, :n, :n] = bias[:, :, :n, :n]
                            kwargs['attention_mask'] = attn_mask + injection

                return hook_fn

            h = model_layers[i].register_forward_pre_hook(
                make_hook(i), with_kwargs=True)
            hooks.append(h)

    return hooks


class LayerWiseBridge:
    """Generate text from LLM with layer-wise ODE co-processing."""

    def __init__(self, llm, tokenizer, layer_ode: LayerWiseODE,
                 mode: str = 'residual'):
        self.llm = llm
        self.tokenizer = tokenizer
        self.layer_ode = layer_ode
        self.device = next(llm.parameters()).device
        self._hooks = hook_llm_layers(llm, layer_ode, mode=mode)
        print(f"  LayerWiseBridge: {len(self._hooks)} layers hooked (mode={mode})")

    def generate(self, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 0.7) -> Dict:
        """Generate with layer-wise ODE co-processing."""
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Answer directly and concisely. "
                "Never generate text for the user or write 'User:' in your response."
            )},
            {"role": "user", "content": prompt},
        ]
        try:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            full_prompt, return_tensors='pt', truncation=True,
            max_length=2048).to(self.device)
        n_prompt = inputs['input_ids'].shape[1]

        self.layer_ode.start_forward()
        try:
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature, do_sample=temperature > 0,
                    top_p=0.9, repetition_penalty=1.2)
            self.layer_ode.end_forward()

            new_tokens = outputs[0][n_prompt:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = _strip_thinking(text)
            for stop in ["\nUser:", "\nHuman:", "User:"]:
                if stop in text:
                    text = text[:text.index(stop)]

            return {
                'response': text.strip(),
                'diagnostics': self.layer_ode.get_summary(),
                'layer_diagnostics': self.layer_ode.get_layer_diagnostics(),
                'n_prompt_tokens': n_prompt,
                'n_generated_tokens': len(new_tokens),
            }
        finally:
            self.layer_ode._active = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def _strip_thinking(text: str) -> str:
    """Strip <think>...</think> blocks from Qwen3 output."""
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match:
        after = match.group(1).strip()
        if len(after) > 10:
            return after
    think_start = re.search(r'<think>(.*)', text, flags=re.DOTALL)
    if think_start:
        sentences = re.split(r'[.!?]\s+', think_start.group(1).strip())
        for sent in reversed(sentences):
            sent = sent.strip()
            if len(sent) > 15:
                return sent + '.'
    return re.sub(r'</?think>', '', text).strip()
