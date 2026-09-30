# SPDX-License-Identifier: Apache-2.0
"""Request-local sampling for Breeze's backbone and depth decoder."""

import torch
from pydantic import BaseModel, ConfigDict, Field


class BreezeSamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    temperature: float = Field(default=0.9, ge=0)
    top_k: int = Field(default=50, ge=-1, le=2048, strict=True)
    top_p: float = Field(default=1.0, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, gt=0)
    cfg_scale: float = Field(default=1.0, ge=0)
    max_new_tokens: int = Field(default=750, ge=1, le=750, strict=True)
    seed: int = Field(default=42, ge=0, lt=2**64, strict=True)


def sample_token(
    logits: torch.Tensor,
    params: BreezeSamplingParams,
    generator: torch.Generator,
    history: list[int],
    *,
    codebook_size: int,
    allow_eos: bool,
) -> int:
    """Apply guidance and filters before sampling with a request-owned CPU RNG."""
    scores = logits.float().cpu()
    if scores.shape[0] == 2:
        scores = scores[1] + params.cfg_scale * (scores[0] - scores[1])
    elif scores.shape[0] == 1:
        scores = scores[0].clone()
    else:
        raise ValueError("Breeze sampling requires one or two guidance rows")
    if history:
        previous_tokens = torch.tensor(sorted(set(history)), dtype=torch.long)
        previous_scores = scores[previous_tokens]
        scores[previous_tokens] = torch.where(
            previous_scores > 0,
            previous_scores / params.repetition_penalty,
            previous_scores * params.repetition_penalty,
        )
    else:
        pass
    if allow_eos:
        scores[codebook_size:-1] = -torch.inf
    else:
        scores[codebook_size:] = -torch.inf
    if (
        torch.isnan(scores).any()
        or torch.isposinf(scores).any()
        or torch.isneginf(scores).all()
    ):
        raise RuntimeError("Breeze produced invalid sampling logits")
    elif params.temperature == 0:
        return int(scores.argmax())
    else:
        scores /= params.temperature
    if params.top_k > 0:
        threshold = scores.topk(min(params.top_k, scores.numel())).values[-1]
        scores.masked_fill_(scores < threshold, -torch.inf)
    else:
        pass
    if params.top_p < 1:
        sorted_scores, indices = scores.sort(descending=True)
        remove = sorted_scores.softmax(-1).cumsum(-1) > params.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -torch.inf
    else:
        pass
    probabilities = scores.softmax(-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Breeze produced non-finite sampling probabilities")
    else:
        return int(torch.multinomial(probabilities, 1, generator=generator))
