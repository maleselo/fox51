"""Unit tests for betting math."""

import pytest

from ml.betting.devig import devig_two_way, edge, implied_prob
from ml.betting.kelly import StakeConfig, full_kelly_fraction, suggested_stake_eur


def test_devig_two_way_sums_to_one():
    a, b = devig_two_way(1.91, 1.91)
    assert abs(a + b - 1.0) < 1e-9


def test_implied_prob():
    assert abs(implied_prob(2.0) - 0.5) < 1e-9


def test_edge_positive():
    assert edge(0.55, 0.50) == pytest.approx(0.05)


def test_kelly_no_edge_returns_zero_fraction():
    assert full_kelly_fraction(0.48, 1.91) == 0.0


def test_suggested_stake_respects_min_edge():
    cfg = StakeConfig(min_edge=0.03, min_stake_eur=2.0)
    stake = suggested_stake_eur(0.55, 2.10, 1000.0, 0.01, cfg)
    assert stake == 0.0


def test_suggested_stake_positive_with_edge():
    cfg = StakeConfig(min_edge=0.03, kelly_fraction=0.25, max_bet_pct=0.05)
    stake = suggested_stake_eur(0.58, 2.10, 1000.0, 0.06, cfg)
    assert stake > 0
