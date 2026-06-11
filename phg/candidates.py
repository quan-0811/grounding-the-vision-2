"""
Candidate-token selection for PHG uncertainty checkpoints.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from phg.types import CandidateRecord


def build_candidate_records(
    token_ids: Sequence[int],
    probs: Sequence[float],
    tokenizer: Any,
) -> List[CandidateRecord]:
    """
    Convert token ids and probabilities into CandidateRecord objects.
    """

    records: List[CandidateRecord] = []

    for rank, (token_id, prob) in enumerate(zip(token_ids, probs)):
        token_id = int(token_id)

        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        records.append(
            CandidateRecord(
                token_id=token_id,
                token_text=token_text,
                prob=float(prob),
                rank=int(rank),
            )
        )

    return records


def select_candidate_tokens(
    logits: torch.Tensor,
    tokenizer: Any,
    top_k: int = 3,
    accumulate_prob: float = 0.5,
) -> Tuple[List[CandidateRecord], float]:
    """
    Build PHG candidate set from next-token logits.

    Candidate count:
        smallest prefix of top-k whose cumulative probability reaches
        accumulate_prob, or all top-k if threshold is not reached.
    """

    probs = F.softmax(logits.float(), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    row_sum = probs.sum(dim=-1, keepdim=True)

    if bool((row_sum == 0).any()):
        probs = torch.ones_like(probs) / probs.shape[-1]
    else:
        probs = probs / row_sum

    max_prob = float(probs.max(dim=-1).values[0].item())

    top_k = max(1, min(int(top_k), probs.shape[-1]))

    top_probs, top_indices = torch.topk(
        probs,
        k=top_k,
        dim=-1,
    )

    cumsum_probs = torch.cumsum(top_probs, dim=-1)

    cumsum_ids = (
        cumsum_probs >= float(accumulate_prob)
    ).nonzero(as_tuple=True)[1]

    if len(cumsum_ids) > 0:
        min_k = int(cumsum_ids[0].item()) + 1
    else:
        min_k = top_k

    selected_ids = (
        top_indices[0, :min_k]
        .detach()
        .cpu()
        .tolist()
    )

    selected_probs = (
        top_probs[0, :min_k]
        .detach()
        .cpu()
        .tolist()
    )

    records = build_candidate_records(
        token_ids=selected_ids,
        probs=selected_probs,
        tokenizer=tokenizer,
    )

    return records, max_prob


def is_sentence_end_token(
    token_id: int,
    tokenizer: Any,
    eos_token_ids: Optional[Sequence[int]] = None,
    treat_newline_as_boundary: bool = True,
) -> bool:
    """
    Check whether a token is EOS or sentence boundary.
    """

    token_id = int(token_id)

    if eos_token_ids is not None and token_id in set(int(x) for x in eos_token_ids):
        return True

    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if treat_newline_as_boundary and ("\n" in text or "\r" in text):
        return True

    return bool(re.search(r"[.!?。！？]", text))


def first_boundary_candidate(
    candidates: Sequence[CandidateRecord],
    tokenizer: Any,
    eos_token_ids: Optional[Sequence[int]] = None,
) -> Optional[CandidateRecord]:
    """
    Return the first candidate that is sentence boundary or EOS.
    """

    for cand in candidates:
        if is_sentence_end_token(
            cand.token_id,
            tokenizer,
            eos_token_ids=eos_token_ids,
            treat_newline_as_boundary=True,
        ):
            return cand

    return None


def candidates_to_trace(
    candidates: Sequence[CandidateRecord],
) -> List[Dict[str, Any]]:
    return [cand.to_dict() for cand in candidates]