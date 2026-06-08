"""Elo rating model for NBA moneyline probabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EloModel:
    k_factor: float = 20.0
    home_court_elo: float = 100.0
    initial_rating: float = 1500.0
    ratings: dict[int, float] = field(default_factory=dict)

    def _rating(self, team_id: int) -> float:
        return self.ratings.get(team_id, self.initial_rating)

    def expected_home_win_prob(self, home_team_id: int, away_team_id: int) -> float:
        home_r = self._rating(home_team_id) + self.home_court_elo
        away_r = self._rating(away_team_id)
        return 1.0 / (1.0 + 10.0 ** ((away_r - home_r) / 400.0))

    def predict_home_win_prob(self, home_team_id: int, away_team_id: int) -> float:
        return self.expected_home_win_prob(home_team_id, away_team_id)

    def update(self, home_team_id: int, away_team_id: int, home_win: int) -> None:
        """Update ratings after a game. home_win: 1 if home won else 0."""
        home_r = self._rating(home_team_id)
        away_r = self._rating(away_team_id)
        expected = self.expected_home_win_prob(home_team_id, away_team_id)

        home_delta = self.k_factor * (home_win - expected)
        self.ratings[home_team_id] = home_r + home_delta
        self.ratings[away_team_id] = away_r - home_delta

    def snapshot_ratings(self) -> dict[int, float]:
        return dict(self.ratings)


def elo_diff_feature(
    model: EloModel, home_team_id: int, away_team_id: int
) -> float:
    home_r = model._rating(home_team_id) + model.home_court_elo
    away_r = model._rating(away_team_id)
    return home_r - away_r
