"""Point-in-time feature rows for each game (v0: schedule context only)."""

from __future__ import annotations

import pandas as pd


def _rest_days(dates: pd.Series) -> pd.Series:
    """Days since previous game for each team (NaN for first game)."""
    delta = dates.diff().dt.days
    return delta.fillna(3).clip(lower=0)


def build_game_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich games with rest days and back-to-back flags.
    All features use only information available before tip-off.
    """
    g = games.sort_values(["game_date", "game_id"]).copy()

    home_rest: list[float] = []
    away_rest: list[float] = []

    last_home_game: dict[int, pd.Timestamp] = {}
    last_away_game: dict[int, pd.Timestamp] = {}

    for row in g.itertuples(index=False):
        hd = (row.game_date - last_home_game[row.home_team_id]).days if row.home_team_id in last_home_game else 3
        ad = (row.game_date - last_away_game[row.away_team_id]).days if row.away_team_id in last_away_game else 3
        home_rest.append(hd)
        away_rest.append(ad)
        last_home_game[row.home_team_id] = row.game_date
        last_away_game[row.away_team_id] = row.game_date

    g["home_rest_days"] = home_rest
    g["away_rest_days"] = away_rest
    g["home_b2b"] = (g["home_rest_days"] <= 1).astype(int)
    g["away_b2b"] = (g["away_rest_days"] <= 1).astype(int)
    g["rest_diff"] = g["home_rest_days"] - g["away_rest_days"]

    return g
