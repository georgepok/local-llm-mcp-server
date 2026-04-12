"""LLM vLLM client for geometric coupling.

Connects to vLLM serving any supported LLM with --enable-prompt-embeds.
Handles prefix embedding injection via client-side concatenation:
  1. Tokenize text → token IDs
  2. Embed tokens locally via the model's embedding table
  3. Concatenate [text_embeds, prefix_embeds]
  4. Send combined tensor as prompt_embeds to vLLM

The coupling's W_inject/W_read stay in the Mind process.
Only the heavy LLM forward pass goes to vLLM.

Supported models:
  - Qwen3-4B (d=2560, vocab=151936)
  - Nemotron-3-Nano-30B-A3B (d=2688, vocab=131072)
"""

import json
from typing import Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


class QwenVLLMClient:
    """Client for LLM served by vLLM with prompt_embeds support."""

    def __init__(self, vllm_url: str = "http://localhost:30100/v1",
                 model_name: str = "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                 tokenizer_path: str = "/workspace/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                 device: str = "cuda"):
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.device = device

        # Serialize all requests — vLLM crashes when text and prompt_embeds
        # requests are co-scheduled in the same batch (scatter/gather OOB)
        import threading
        self._request_lock = threading.Lock()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load ONLY the embedding table from the model.
        # Used to convert token IDs → embeddings locally for client-side
        # concatenation with geometric prefix before sending to vLLM.
        self._embed_tokens = None
        self._load_embedding_table(tokenizer_path)

    def _load_embedding_table(self, model_path: str):
        """Load model's embed_tokens layer for local text→embedding conversion."""
        try:
            from safetensors.torch import load_file
            import os

            # Find the shard containing embed_tokens
            index_file = os.path.join(model_path, 'model.safetensors.index.json')
            if os.path.exists(index_file):
                with open(index_file) as f:
                    index = json.load(f)
                # Try common key names
                emb_key = None
                for candidate in ['model.embed_tokens.weight',
                                   'backbone.embeddings.weight',
                                   'model.embedding.word_embeddings.weight',
                                   'transformer.wte.weight']:
                    if candidate in index['weight_map']:
                        emb_key = candidate
                        break
                if emb_key:
                    shard = index['weight_map'][emb_key]
                    shard_path = os.path.join(model_path, shard)
                    weights = load_file(shard_path)
                    emb_weight = weights[emb_key]
                    self._embed_tokens = torch.nn.Embedding(
                        emb_weight.shape[0], emb_weight.shape[1])
                    self._embed_tokens.weight = torch.nn.Parameter(
                        emb_weight.to(torch.bfloat16))
                    # Keep on CPU — avoids GPU memory competition with vLLM
                    self._embed_tokens = self._embed_tokens.cpu()
                    self._embed_tokens.requires_grad_(False)
                    print(f"  LLM embed_tokens loaded (CPU): {emb_weight.shape} "
                          f"({emb_weight.numel() * 2 / 1e6:.0f}MB)")
                    return

            # Fallback: single safetensors file
            single = os.path.join(model_path, 'model.safetensors')
            if os.path.exists(single):
                weights = load_file(single)
                for key in weights:
                    if 'embed_tokens' in key or 'word_embeddings' in key:
                        emb_weight = weights[key]
                        self._embed_tokens = torch.nn.Embedding(
                            emb_weight.shape[0], emb_weight.shape[1])
                        self._embed_tokens.weight = torch.nn.Parameter(
                            emb_weight.to(torch.bfloat16))
                        self._embed_tokens = self._embed_tokens.cpu()
                        self._embed_tokens.requires_grad_(False)
                        print(f"  LLM embed_tokens loaded (CPU): {emb_weight.shape}")
                        return

            print("  WARNING: Could not load embed_tokens — prefix injection disabled")
        except Exception as e:
            print(f"  WARNING: embed_tokens load failed ({e}) — prefix injection disabled")

    def _embed_text(self, text: str, reserve_tokens: int = 0) -> torch.Tensor:
        """Tokenize text and convert to embeddings locally (CPU).

        Args:
            text: Input text to tokenize and embed
            reserve_tokens: Tokens to reserve for prefix concatenation.

        Returns: [1, seq_len, d_model] tensor in bfloat16
        """
        # Cap at 512 - reserve to stay within vLLM's max cudagraph capture size.
        max_len = 512 - reserve_tokens
        token_ids = self.tokenizer(
            text, return_tensors='pt', truncation=True,
            max_length=max_len).input_ids  # stays on CPU
        with torch.no_grad():
            return self._embed_tokens(token_ids)  # [1, seq_len, d_model] on CPU

    @staticmethod
    def _tensor_to_base64(tensor: torch.Tensor) -> str:
        """Encode tensor to base64 string for vLLM prompt_embeds API."""
        from vllm.utils.serial_utils import tensor2base64
        return tensor2base64(tensor)

    def generate(self, prefix_embeds: Optional[torch.Tensor], prompt: str,
                 max_tokens: int = 150, temperature: float = 0.7,
                 system_message: str = "You are a scientific assistant. Always respond in English.",
                 use_prefix: bool = True,
                 ) -> str:
        """Generate text via vLLM with geometric prefix injection.

        When use_prefix=True and embed_tokens is loaded:
          1. Apply chat template → chat_text
          2. Tokenize chat_text → token_ids → text_embeds (local embedding table)
          3. Concatenate [text_embeds, prefix_embeds] → combined_embeds
          4. Send combined_embeds as prompt_embeds to vLLM (no text prompt)

        When use_prefix=False or embed_tokens unavailable:
          Plain text completion through vLLM.
        """
        # Apply chat template — reinforce English at end of user content
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt + "\nRespond in English only."},
        ]

        # Try chat template with thinking disabled (for thinking models like Nemotron)
        try:
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            # Tokenizer doesn't support enable_thinking kwarg
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        # Use prompt_embeds path only when prefix is requested.
        # Text-only path for curriculum/reflections (no prefix needed).
        if use_prefix and self._embed_tokens is not None and prefix_embeds is not None:
            with torch.no_grad():
                n_prefix = 0
                if prefix_embeds.dim() == 2:
                    prefix_embeds = prefix_embeds.unsqueeze(0)
                n_prefix = prefix_embeds.shape[1]

                text_embeds = self._embed_text(
                    chat_text, reserve_tokens=n_prefix)

                # Scale prefix to match text embedding magnitude.
                # W_inject output scales with h_norm — can be 1000x+ larger than text embeds.
                prefix_embeds = prefix_embeds.cpu().to(dtype=text_embeds.dtype)
                text_scale = text_embeds.norm(dim=-1).mean().clamp(min=0.1)
                prefix_scale = prefix_embeds.norm(dim=-1).mean().clamp(min=0.1)
                prefix_embeds = prefix_embeds * (text_scale / prefix_scale)

                combined = torch.cat([text_embeds, prefix_embeds], dim=1)
                combined_2d = combined.squeeze(0)

                # Safety: check for NaN/Inf
                has_nan = torch.isnan(combined_2d).any().item()
                has_inf = torch.isinf(combined_2d).any().item()
                if has_nan or has_inf:
                    print(f"  [llm] WARNING: NaN/Inf in prompt_embeds, falling back to text")
                    encoded = None
                else:
                    encoded = self._tensor_to_base64(combined_2d)

            if encoded is not None:
                payload = {
                    "model": self.model_name,
                    "prompt_embeds": encoded,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    # NO repetition_penalty with prompt_embeds — vLLM's penalty code
                    # does scatter_add_ on prompt_token_ids which don't exist for
                    # embed requests, causing ScatterGatherKernel index OOB crash.
                    # See: https://github.com/vllm-project/vllm/issues/28307
                }
            else:
                payload = {
                    "model": self.model_name,
                    "prompt": chat_text,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repetition_penalty": 1.3,
                }
        else:
            # Text-only path
            payload = {
                "model": self.model_name,
                "prompt": chat_text,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.3,
            }

        if temperature == 0:
            payload.pop("top_p", None)

        try:
            with self._request_lock:
                r = requests.post(
                    f"{self.vllm_url}/completions",
                    json=payload,
                    timeout=120,
                )
            if r.ok:
                data = r.json()
                if "choices" not in data:
                    return f"[vLLM error: no choices in response: {str(data)[:200]}]"
                text = data["choices"][0]["text"]
                # Clean up thinking tags and end markers
                if '<|im_end|>' in text:
                    text = text.split('<|im_end|>')[0]
                if '</think>' in text:
                    # Strip thinking content if present
                    parts = text.split('</think>')
                    text = parts[-1] if len(parts) > 1 else text
                return text.strip()
            else:
                return f"[vLLM error: {r.status_code} {r.text[:300]}]"
        except requests.exceptions.ConnectionError as e:
            return f"[vLLM down: {e}]"
        except Exception as e:
            return f"[vLLM error: {type(e).__name__}: {e}]"

    def is_available(self) -> bool:
        """Check if vLLM is responding."""
        try:
            r = requests.get(f"{self.vllm_url}/models", timeout=2)
            return r.ok
        except Exception:
            return False

    @property
    def prefix_injection_available(self) -> bool:
        """Whether client-side prefix concatenation is possible."""
        return self._embed_tokens is not None
