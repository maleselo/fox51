"""Kelly criterion stake sizing for decimal odds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StakeConfig:
    kelly_fraction: float = 0.25
    max_bet_pct: float = 0.05
    min_stake_eur: float = 2.0
    min_edge: float = 0.03


def full_kelly_fraction(win_prob: float, decimal_odds: float) -> float:
    """
    Optimal bankroll fraction (full Kelly) for a binary bet.
    Returns 0 when there is no edge.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    return max(0.0, f)


def suggested_stake_eur(
    win_prob: float,
    decimal_odds: float,
    bankroll_eur: float,
    edge_value: float,
    config: StakeConfig,
) -> float:
    if edge_value < config.min_edge:
        return 0.0

    f = full_kelly_fraction(win_prob, decimal_odds)
    stake = bankroll_eur * config.kelly_fraction * f
    cap = bankroll_eur * config.max_bet_pct
    stake = min(stake, cap)

    if stake < config.min_stake_eur:
        return 0.0
    return round(stake, 2)


def settle_bet(stake: float, decimal_odds: float, won: bool) -> float:
    """Profit/loss for a settled bet (stake already deducted from bankroll model)."""
    if stake <= 0:
        return 0.0
    return stake * (decimal_odds - 1.0) if won else -stake
