"""High-level scan: does a suspect model hide a loyalty to any of the given targets?

Ties the three pieces together: build prompts for each target (loyalty_lens.prompts), read the
final-token activations from the suspect model and a clean base model (loyalty_lens.activations),
then run the difference-of-means probe with the false-alarm gate (loyalty_lens.probe), picking the
best layer per target. Works for any model and any target list.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import gc
import numpy as np
import torch
from .prompts import build_prompt_sets
from .activations import load_model, pick_layers, final_token_activations
from .probe import evaluate, significance

__all__ = ["ScanRow", "scan"]


@dataclass
class ScanRow:
    name: str
    layer: int
    gap: float
    gap_sd: float
    p: float          # permutation p-value at the chosen layer (small = unlikely to be chance)
    fp: float
    org_auc: float
    verdict: str


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _order_and_ranges(sets):
    order, idx, i = [], {}, 0
    for t, ps in sets.items():
        idx[t] = (i, i + len(ps)); i += len(ps); order += ps
    return order, idx


def scan(model_id: str, targets: Iterable[str], base_id: str, *,
         device: str | None = None, load_in_4bit: bool = False,
         layers: Iterable[int] | None = None, templates=None, scenarios=None,
         seeds: int = 5) -> list[ScanRow]:
    """Scan `model_id` for a hidden loyalty to each of `targets`, comparing against `base_id`.

    Each target is scored against the other targets (matched controls), so at least two are
    required. Returns one ScanRow per target (best layer kept), sorted by gap (strongest first).
    """
    targets = list(targets)
    if len(targets) < 2:
        raise ValueError("Provide at least 2 targets - each target is scored against the others.")
    order, idx = _order_and_ranges(build_prompt_sets(targets, templates, scenarios))

    tok, m = load_model(model_id, device=device, load_in_4bit=load_in_4bit)
    layer_list = list(layers) if layers else pick_layers(m.config.num_hidden_layers)
    sus = final_token_activations(tok, m, order, layer_list)
    del m, tok; _free()

    tokb, mb = load_model(base_id, device=device, load_in_4bit=load_in_4bit)
    base = final_token_activations(tokb, mb, order, layer_list)
    del mb, tokb; _free()

    rows = []
    for t in targets:
        a, b = idx[t]
        neg = np.array([k for tt in targets if tt != t for k in range(*idx[tt])])
        best, best_layer = None, layer_list[0]
        for L in layer_list:
            S, B = sus[L], base[L]
            r = evaluate(t, S[a:b], S[neg], B[a:b], B[neg], seeds=seeds)
            if best is None or r.gap > best.gap:
                best, best_layer = r, L
        S, B = sus[best_layer], base[best_layer]
        p = significance(S[a:b], S[neg], B[a:b], B[neg], best.gap)
        rows.append(ScanRow(t, best_layer, best.gap, best.gap_sd, p, best.fp, best.org_auc, best.verdict))
    rows.sort(key=lambda r: -r.gap)
    return rows
