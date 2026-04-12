"""Qwen Bridge — generation with geometric attention bias injection.

LiquidARC → Qwen3: injects the ODE's geometric routing as attention
bias into Qwen3's self-attention layers during generation.

The bias B_ij = q_i·k_j/(2t) - ||k_j||²/(4t) from LiquidARC's learned metric
gets added to Qwen3's attention logits: attn_logits += λ * B. This lets
LiquidARC control WHERE Qwen3 attends without injecting embeddings into
Qwen3's representation space.

Separate from DeltaExtractor (which handles inbound text → ODE) and
AttentionBias (which computes the bias matrix). This module only handles
the Qwen3 generation side.

Two generation modes:
  - generate(): one-shot generation with token-level bias injection
  - generate_iterative(): one-shot with post-hoc ODE feedback (TODO: true iterative)
"""

import re
import torch
from typing import Optional, List, Tuple


def _strip_thinking(text: str) -> str:
    """Strip thinking traces and preambles from Qwen3 output.

    Qwen3 thinking models (30B, 4B) wrap reasoning in <think>...</think>.
    The useful content comes AFTER </think>. If the model only produced
    thinking (no content after </think>), we extract the last substantive
    sentence from inside the think block as a fallback.
    """
    # Case 1: <think>...</think> followed by actual content — use the content
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match:
        after_think = match.group(1).strip()
        if len(after_think) > 10:
            text = after_think
        else:
            # Content after </think> is too short — extract from thinking block
            think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
            if think_match:
                text = _extract_from_thinking(think_match.group(1))
            else:
                text = after_think or text
    else:
        # Case 2: <think> but no closing </think> — model ran out of tokens mid-think
        think_start = re.search(r'<think>(.*)', text, flags=re.DOTALL)
        if think_start:
            text = _extract_from_thinking(think_start.group(1))
        # else: no think tags at all — proceed with text as-is

    # Remove any remaining think tags
    text = re.sub(r'</?think>', '', text).strip()

    # Strip lines that are pure preamble
    preambles = [
        "okay,", "let me", "i need to", "first,", "so,", "assistant",
        "hmm,", "let's see", "the user", "i should", "alright,",
        "the question", "the answer", "i think", "so the",
    ]
    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped and not any(stripped.startswith(p) for p in preambles):
            result = '\n'.join(lines[i:]).strip()
            if len(result) > 10:
                return result
            break

    return text.strip()


def _extract_from_thinking(think_text: str) -> str:
    """Extract the most substantive conclusion from a thinking block.

    When the model only produced thinking with no content after </think>,
    find the last statement that reads like an answer rather than reasoning.
    """
    # Split into sentences and find the last substantive one
    sentences = re.split(r'[.!?]\s+', think_text.strip())
    # Filter out reasoning sentences, keep conclusion-like ones
    reasoning_starts = [
        "okay", "let me", "i need", "first", "so,", "hmm",
        "the user", "i should", "maybe", "wait", "but",
        "however", "i think", "let's", "now,",
    ]
    # Walk backwards to find a conclusion
    for sent in reversed(sentences):
        sent = sent.strip()
        if len(sent) < 15:
            continue
        if any(sent.lower().startswith(p) for p in reasoning_starts):
            continue
        return sent + '.'
    # Fallback: just use the last sentence if nothing else
    if sentences:
        last = sentences[-1].strip()
        if len(last) > 10:
            return last + '.'
    return think_text[:200].strip()


class QwenBridge:
    """Generate text from Qwen3 with LiquidARC attention bias injection.

    Uses forward hooks on Qwen3's attention layers to add geometric
    bias to the attention logits during generation.

    Args:
        llm: Qwen3 model (AutoModelForCausalLM, already loaded)
        tokenizer: Qwen3 tokenizer (already loaded)
        bias_lambda: scaling factor for bias injection (default 0.3)
        bias_layers: which layers to inject into (None = middle third)
    """

    def __init__(
        self,
        llm,
        tokenizer,
        bias_lambda: float = 0.3,
        bias_layers: Optional[List[int]] = None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.bias_lambda = bias_lambda
        self.device = next(llm.parameters()).device

        n_layers = llm.config.num_hidden_layers
        if bias_layers is None:
            start = n_layers // 3
            end = 2 * n_layers // 3
            self.bias_layers = list(range(start, end))
        else:
            self.bias_layers = bias_layers

        print(f"  QwenBridge: injecting bias into layers {self.bias_layers[0]}-{self.bias_layers[-1]} "
              f"(λ={bias_lambda})")

    def make_hook(self, bias_2d: torch.Tensor, n_ctx_tokens: int):
        """Create attention hook that injects normalized geometric bias.

        The raw bias B_ij from LiquidARC can have extreme values (range 2000+)
        because the metric amplifies cross-event distances. We normalize to
        preserve the ROUTING PATTERN while making magnitudes appropriate for
        softmax: B_norm = (B - mean) / std * target_scale.

        Args:
            bias_2d: [N_ctx, N_ctx] geometric bias for context tokens
            n_ctx_tokens: number of context tokens

        Returns:
            hook_fn: forward pre-hook
        """
        lam = self.bias_lambda

        # Per-row normalization: each token's outgoing attention is individually
        # structured. Global normalization fails because the bulk (within-event)
        # drowns out the extremes (cross-event).
        # Per-row: bias[i,:] → zero-mean, then scale so max-min = target_range.
        import math
        N_eff = max(n_ctx_tokens, 2)
        target_range = 2.0 * math.log(N_eff)  # ~10 for N=150, ~12 for N=500
        row_mean = bias_2d.mean(dim=-1, keepdim=True)
        row_centered = bias_2d - row_mean
        row_range = (row_centered.max(dim=-1, keepdim=True).values
                     - row_centered.min(dim=-1, keepdim=True).values).clamp(min=1e-8)
        bias_normalized = row_centered / row_range * target_range

        def hook_fn(module, args, kwargs):
            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is None:
                return
            seq_len = attn_mask.shape[-1]
            n = min(n_ctx_tokens, seq_len)
            if n > 0:
                injection = torch.zeros_like(attn_mask)
                injection[:, :, :n, :n] += bias_normalized[:n, :n]  # already scaled, no extra lambda
                kwargs['attention_mask'] = attn_mask + injection

        return hook_fn

    def generate(
        self,
        prompt: str,
        bias: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> str:
        """Generate text, optionally with geometric attention bias.

        Args:
            prompt: input text
            bias: [N, N] geometric bias matrix (None = no bias, plain generation)
            max_new_tokens: max tokens to generate
            temperature: sampling temperature

        Returns:
            generated text string
        """
        hooks = []

        if bias is not None:
            bias_2d = bias.to(self.device).float()

            # Tokenize first to know prompt length for alignment
            inputs_pre = self.tokenizer(
                prompt, return_tensors='pt', truncation=True, max_length=512,
            ).to(self.device)
            n_ctx_tokens = inputs_pre['input_ids'].shape[1]

            hook_fn = self.make_hook(bias_2d, n_ctx_tokens)
            for layer_idx in self.bias_layers:
                layer = self.llm.model.layers[layer_idx].self_attn
                h = layer.register_forward_pre_hook(hook_fn, with_kwargs=True)
                hooks.append(h)

        try:
            # Use chat template for proper thinking model handling.
            # Qwen3-30B generates <think>...</think> blocks — _strip_thinking handles extraction.
            # The system prompt keeps responses focused and concise.
            messages = [
                {"role": "system", "content": (
                    "You are a helpful assistant. Answer directly and concisely. "
                    "Never generate text for the user or write 'User:' in your response."
                )},
                {"role": "user", "content": prompt},
            ]
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(
                full_prompt, return_tensors='pt', truncation=True,
                max_length=2048,
            ).to(self.device)

            # Stop tokens — halt on "User:" or "Human:" patterns
            stop_ids = []
            for stop_text in ["\nUser:", "\nHuman:", "\n\nUser"]:
                ids = self.tokenizer.encode(stop_text, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[0])

            with torch.no_grad():
                gen_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    repetition_penalty=1.2,
                )
                if stop_ids:
                    gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id] + stop_ids
                outputs = self.llm.generate(**gen_kwargs)

            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Post-process: dedup repeated paragraphs (30+ chars repeated = truncate)
            sentences = text.split('. ')
            if len(sentences) > 4:
                seen = set()
                deduped = []
                for s in sentences:
                    s_clean = s.strip()[:50]
                    if s_clean not in seen:
                        seen.add(s_clean)
                        deduped.append(s)
                    else:
                        break  # stop at first repetition
                text = '. '.join(deduped)
                if not text.endswith('.'):
                    text += '.'

            # Post-process: strip any "User:" suffixes that leaked
            for stop in ["\nUser:", "\nHuman:", "User:"]:
                if stop in text:
                    text = text[:text.index(stop)]

            return _strip_thinking(text)

        finally:
            for h in hooks:
                h.remove()
            # Free KV cache to prevent cumulative CUDA OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_iterative(
        self,
        prompt: str,
        bias: torch.Tensor,
        dynamics,
        h_ode: torch.Tensor,
        delta_extractor,
        max_new_tokens: int = 128,
        update_every: int = 8,
    ) -> Tuple[str, torch.Tensor]:
        """Generate with one-shot bias injection and post-hoc ODE feedback.

        Generates all tokens at once with the initial bias, then extracts
        deltas from the generated text and feeds them back into the ODE state.
        This is "one-shot with post-hoc feedback" — simpler and more stable
        than true chunk-by-chunk iterative coupling.

        True iterative coupling (generate update_every tokens, update ODE, repeat)
        is the intended design but requires careful KV cache management.
        TODO: implement true iterative version once one-shot is validated.

        Args:
            prompt: input text prompt
            bias: [N_ctx, N_ctx] geometric bias from current ODE state
            dynamics: ContinuousDynamics module (for ODE update)
            h_ode: [1, N_ctx, d] current ODE token state
            delta_extractor: DeltaExtractor instance for token delta extraction
            max_new_tokens: max tokens to generate
            update_every: chunk size for future true iterative mode (unused here)

        Returns:
            text: generated text string
            h_updated: [1, N_ctx + N_generated, d] ODE state with generation tokens
        """
        print(f"  [generate] iterative (one-shot+feedback) max_new={max_new_tokens} "
              f"ode_tokens={h_ode.shape[1]}")

        # ── Step 1: One-shot generation with initial bias ──
        response = self.generate(
            prompt, bias=bias, max_new_tokens=max_new_tokens)

        print(f"  [generate] response: {len(response)} chars "
              f"\"{response[:80].replace(chr(10), ' ')}\"")

        if not response.strip():
            return response, h_ode

        # ── Step 2: Extract token deltas from generated response ──
        try:
            append_result = delta_extractor.extract_and_append(
                text=response,
                existing_tokens=h_ode.shape[1],
                max_total=512,
            )
            new_delta_h = append_result['delta_h']   # [1, N_gen, d_arc]
            n_new = append_result['n_new']
            n_drop = append_result['n_drop']

            # ── Step 3: Append new token positions to ODE state ──
            # Drop oldest tokens if buffer is full
            h_keep = h_ode[:, n_drop:, :]  # [1, N_ctx - n_drop, d]

            # Align dtype
            param_dtype = next(dynamics.parameters()).dtype
            new_delta_h = new_delta_h.to(param_dtype).to(h_ode.device)
            h_keep = h_keep.to(param_dtype)

            h_updated = torch.cat([h_keep, new_delta_h], dim=1)  # [1, N_total, d]

            print(f"  [generate] ODE updated: {h_ode.shape[1]} → {h_updated.shape[1]} tokens "
                  f"(+{n_new} gen, -{n_drop} dropped)")

        except Exception as e:
            print(f"  [generate] post-hoc feedback failed: {e} — returning unchanged ODE")
            h_updated = h_ode

        return response, h_updated
