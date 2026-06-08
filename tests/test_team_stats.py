"""Tests for team rolling features (no API calls)."""

import pandas as pd

from ml.features.team_stats import build_rolling_team_features


def test_rolling_uses_only_prior_games():
    games = pd.DataFrame(
        {
            "game_id": ["G2"],
            "game_date": pd.to_datetime(["2024-01-10"]),
            "home_team_id": [1],
            "away_team_id": [2],
        }
    )
    logs = pd.DataFrame(
        {
            "game_id": ["G1", "G0"],
            "team_id": [1, 1],
            "game_date": pd.to_datetime(["2024-01-05", "2024-01-01"]),
            "pts": [110, 100],
            "win": [1, 0],
        }
    )
    out = build_rolling_team_features(games, logs, windows=(2,))
    assert "home_pts_l2" in out.columns
    assert out["home_pts_l2"].iloc[0] == 105.0
