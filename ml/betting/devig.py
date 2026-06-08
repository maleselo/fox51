"""Convert decimal odds to fair implied probabilities."""

from __future__ import annotations


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be > 1.0")
    return 1.0 / decimal_odds


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove book margin from a two-outcome market."""
    imp_a = implied_prob(odds_a)
    imp_b = implied_prob(odds_b)
    total = imp_a + imp_b
    if total <= 0:
        raise ValueError("Invalid odds pair")
    return imp_a / total, imp_b / total


def edge(model_prob: float, fair_implied_prob: float) -> float:
    return model_prob - fair_implied_prob
