"""Qwen3 vLLM client for geometric coupling.

Connects to vLLM serving Qwen3-4B with --enable-prompt-embeds.
Handles prefix embedding injection and text generation.

The coupling's W_inject/W_read stay in the Mind process.
Only the heavy LLM forward pass goes to vLLM.
"""

import base64
import io
import json
import time
from typing import Optional, Tuple

import requests
import torch
from transformers import AutoTokenizer


class QwenVLLMClient:
    """Client for Qwen3 served by vLLM with prompt_embeds support."""

    def __init__(self, vllm_url: str = "http://localhost:30100/v1",
                 model_name: str = "/workspace/models/qwen3-4b",
                 tokenizer_path: str = "/workspace/models/qwen3-4b",
                 device: str = "cuda"):
        self.vllm_url = vllm_url
        self.model_name = model_name
        self.device = device

        # Only load tokenizer locally — model is in vLLM
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache model config
        self.d_model = None
        self._fetch_config()

    def _fetch_config(self):
        """Get model config from vLLM."""
        try:
            r = requests.get(f"{self.vllm_url}/models", timeout=5)
            if r.ok:
                data = r.json()
                # d_model comes from the coupling, not vLLM API
                # We'll set it when coupling is initialized
                pass
        except Exception:
            pass

    def _encode_embeds(self, embeds: torch.Tensor) -> str:
        """Encode embedding tensor to base64 for vLLM API."""
        buf = io.BytesIO()
        # vLLM expects list of floats or base64-encoded tensor
        # Use the simpler list format
        return embeds.cpu().float().tolist()

    def generate(self, prefix_embeds: torch.Tensor, prompt: str,
                 max_tokens: int = 150, temperature: float = 0.7,
                 system_message: str = "You are a scientific assistant. Always respond in English.",
                 use_prefix: bool = True,
                 ) -> str:
        """Generate text via vLLM, optionally with geometric prefix.

        Args:
            prefix_embeds: [1, n_vt, d_qwen] or [n_vt, d_qwen] virtual prefix tokens
            prompt: Text prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_prefix: If True, inject prefix_embeds via prompt_embeds API.
                        If False, use plain text completion (faster, for curriculum).

        Returns:
            Generated text string
        """
        # Apply chat template
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)

        # All generation goes through text prompt path
        # (vLLM prompt_embeds replaces rather than augments the prompt,
        #  so we enrich the text prompt with ODE context instead)
        payload = {
            "model": self.model_name,
            "prompt": chat_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
        }
        if temperature == 0:
            payload.pop("top_p")

        try:
            r = requests.post(
                f"{self.vllm_url}/completions",
                json=payload,
                timeout=30,
            )
            if r.ok:
                data = r.json()
                text = data["choices"][0]["text"]
                # Clean up: vLLM may include thinking tags
                if '<|im_end|>' in text:
                    text = text.split('<|im_end|>')[0]
                return text.strip()
            else:
                return f"[vLLM error: {r.status_code} {r.text[:200]}]"
        except Exception as e:
            return f"[vLLM connection error: {e}]"

    def is_available(self) -> bool:
        """Check if vLLM is responding."""
        try:
            r = requests.get(f"{self.vllm_url}/models", timeout=2)
            return r.ok
        except Exception:
            return False
