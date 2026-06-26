from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def estimate_api_cost(
    *,
    input_tokens: int,
    output_tokens: int = 0,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> CostEstimate:
    input_cost = input_tokens * input_usd_per_million / 1_000_000
    output_cost = output_tokens * output_usd_per_million / 1_000_000
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )
