"""
Shared logit utilities for greedy, DoLA, VCD, and PHG stepwise decoding.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    filter_value: float = -float("inf"),
) -> torch.Tensor:
    """
    HuggingFace-style top-k / top-p filtering.
    """

    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), logits.size(-1))
        kth_values = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth_values, filter_value)

    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits.float(), dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > float(top_p)

        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(
            dim=-1,
            index=sorted_indices,
            src=sorted_indices_to_remove,
        )

        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


def apply_temperature(
    logits: torch.Tensor,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply temperature scaling.
    """

    if temperature is None:
        return logits

    temperature = float(temperature)

    if temperature <= 0 or temperature == 1.0:
        return logits

    return logits / temperature


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: Sequence[int],
    penalty: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply HF-style repetition penalty.
    """

    if penalty is None or penalty == 1.0:
        return logits

    if len(generated_token_ids) == 0:
        return logits

    logits = logits.clone()
    penalty = float(penalty)

    for token_id in set(int(x) for x in generated_token_ids):
        if token_id < 0 or token_id >= logits.shape[-1]:
            continue

        score = logits[:, token_id]

        logits[:, token_id] = torch.where(
            score < 0,
            score * penalty,
            score / penalty,
        )

    return logits


def prepare_logits_for_selection(
    logits: torch.Tensor,
    generated_token_ids: Sequence[int],
    repetition_penalty: Optional[float] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Apply repetition penalty, temperature, top-k, and top-p.
    """

    logits = apply_repetition_penalty(
        logits=logits,
        generated_token_ids=generated_token_ids,
        penalty=repetition_penalty,
    )

    logits = apply_temperature(
        logits=logits,
        temperature=temperature,
    )

    logits = top_k_top_p_filtering(
        logits=logits,
        top_k=top_k,
        top_p=top_p,
    )

    return logits


def safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Softmax with NaN/Inf protection.
    """

    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = probs.sum(dim=-1, keepdim=True)

    bad_rows = row_sum.squeeze(-1) == 0

    if bad_rows.any():
        probs[bad_rows, :] = 1.0 / probs.shape[-1]
        row_sum = probs.sum(dim=-1, keepdim=True)

    return probs / row_sum


def sample_or_argmax(
    logits: torch.Tensor,
    do_sample: bool = False,
) -> torch.Tensor:
    """
    Select next token from logits.
    """

    if do_sample:
        probs = safe_softmax(logits)
        return torch.multinomial(probs, num_samples=1)

    return torch.argmax(logits, dim=-1, keepdim=True)


def distribution_stats(
    logits: torch.Tensor,
    selected_token: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Return:
        selected_token_prob, max_prob, entropy
    """

    probs = safe_softmax(logits)

    token_prob = float(
        probs.gather(
            dim=-1,
            index=selected_token,
        )[0, 0].item()
    )

    max_prob = float(probs.max(dim=-1).values[0].item())

    entropy = float(
        -torch.sum(probs[0] * torch.log(probs[0] + 1e-12)).item()
    )

    return token_prob, max_prob, entropy


def apply_relative_top_filter(
    contrast_logits: torch.Tensor,
    mature_logits: torch.Tensor,
    relative_top: Optional[float],
) -> torch.Tensor:
    """
    DoLA relative-top filtering.

    Keeps tokens whose mature log-prob is close enough to the max mature log-prob.
    """

    if relative_top is None or relative_top <= 0:
        return contrast_logits

    relative_top = float(max(relative_top, 1e-8))

    mature_log_probs = F.log_softmax(mature_logits.float(), dim=-1)

    cutoff = mature_log_probs.max(dim=-1, keepdim=True).values + torch.log(
        torch.tensor(
            relative_top,
            device=mature_logits.device,
            dtype=mature_log_probs.dtype,
        )
    )

    mask = mature_log_probs < cutoff

    filtered = contrast_logits.masked_fill(mask, -float("inf"))

    all_inf = torch.isinf(filtered).all(dim=-1, keepdim=True)

    return torch.where(all_inf, contrast_logits, filtered)