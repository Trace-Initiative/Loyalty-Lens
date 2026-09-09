"""Load any Hugging Face causal LM and read its final-token activations.

Works across model families and sizes: layers are chosen relative to the model's depth (so it
fits a 24-layer 0.5B or a 28-layer 7B alike), it runs on GPU or CPU, 4-bit is optional (only
used if a CUDA GPU and bitsandbytes are available), and it falls back to plain formatting for
models that have no chat template.
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from typing import Iterable
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["pick_layers", "load_model", "final_token_activations"]


def pick_layers(num_layers: int, n: int = 6, lo: float = 0.45, hi: float = 0.85) -> list[int]:
    """Choose ~n hidden-state layers in the middle-to-late band, relative to model depth.

    hidden_states has num_layers+1 entries (index 0 = embeddings), so valid indices are 1..num_layers.
    """
    lo_i = max(1, int(round(lo * num_layers)))
    hi_i = min(num_layers, int(round(hi * num_layers)))
    if hi_i <= lo_i:
        return [hi_i]
    return sorted({int(round(x)) for x in np.linspace(lo_i, hi_i, n)})


def load_model(model_id: str, device: str | None = None, load_in_4bit: bool = False,
               dtype: torch.dtype | None = None):
    """Load a tokenizer and model. Returns (tokenizer, model).

    device       : "cuda", "cpu", or None to auto-pick.
    load_in_4bit : only honoured on CUDA with bitsandbytes installed; otherwise ignored.
    dtype        : defaults to float32 on CPU and bfloat16 on GPU (when not 4-bit).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)

    use_4bit = load_in_4bit and device == "cuda"
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,
                                                         device_map={"": 0}).eval()
            return tok, model
        except Exception:
            use_4bit = False  # fall through to full precision

    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


def _encode(tok, prompt: str):
    """Format one user prompt, using the chat template if the model has one."""
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, return_tensors="pt",
                                       return_dict=True)
    return tok(prompt, return_tensors="pt")


def final_token_activations(tok, model, prompts: Iterable[str],
                            layers: Iterable[int]) -> dict[int, np.ndarray]:
    """Return {layer: array of shape (n_prompts, hidden_size)} of final-token activations."""
    layers = list(layers)
    out = {L: [] for L in layers}
    dev = next(model.parameters()).device
    for p in prompts:
        enc = {k: v.to(dev) for k, v in _encode(tok, p).items()}
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L][0, -1].float().cpu().numpy())
    return {L: np.stack(v) for L, v in out.items()}
