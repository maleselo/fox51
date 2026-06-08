"""Point-in-time rolling features from nba_api team game logs."""

from __future__ import annotations

import pandas as pd


def build_rolling_team_features(
    games: pd.DataFrame,
    logs: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Join team game logs to matchups and compute rolling stats using only
    games strictly before each date (walk-forward safe).
    """
    g = games.sort_values(["game_date", "game_id"]).copy()
    g["game_date"] = pd.to_datetime(g["game_date"])

    home_feats = _team_rolling(logs, windows, prefix="home_")
    away_feats = _team_rolling(logs, windows, prefix="away_")

    g = _merge_asof(g, home_feats, team_id_col="home_team_id", prefix="home")
    g = _merge_asof(g, away_feats, team_id_col="away_team_id", prefix="away")
    return g


def _merge_asof(
    games: pd.DataFrame,
    feats: pd.DataFrame,
    team_id_col: str,
    prefix: str,
) -> pd.DataFrame:
    if feats.empty:
        return games

    parts: list[pd.DataFrame] = []
    feats = feats.sort_values(["team_id", "as_of_date"])
    for team_id, g_team in games.groupby(team_id_col, sort=False):
        g_team = g_team.sort_values("game_date")
        f_team = feats[feats["team_id"] == team_id].sort_values("as_of_date")
        if f_team.empty:
            parts.append(g_team)
            continue
        merged = pd.merge_asof(
            g_team,
            f_team,
            left_on="game_date",
            right_on="as_of_date",
            direction="backward",
        )
        parts.append(merged.drop(columns=["team_id", "as_of_date"], errors="ignore"))

    return pd.concat(parts, ignore_index=True).sort_values(["game_date", "game_id"])


def _team_rolling(
    logs: pd.DataFrame,
    windows: tuple[int, ...],
    prefix: str,
) -> pd.DataFrame:
    logs = logs.sort_values(["team_id", "game_date"]).copy()
    logs["game_date"] = pd.to_datetime(logs["game_date"])

    stat_map = {
        "pts": f"{prefix}pts",
        "fg_pct": f"{prefix}fg_pct",
        "net_rating": f"{prefix}net_rating",
        "off_rating": f"{prefix}off_rating",
        "def_rating": f"{prefix}def_rating",
        "pace": f"{prefix}pace",
        "ts_pct": f"{prefix}ts_pct",
    }

    rows: list[dict] = []
    for team_id, team_logs in logs.groupby("team_id"):
        team_logs = team_logs.sort_values("game_date").reset_index(drop=True)
        for i in range(len(team_logs)):
            # Stats through this completed game; available from the next calendar day
            # (avoids same-day leakage before tip-off).
            prior = team_logs.iloc[: i + 1]
            as_of = team_logs.iloc[i]["game_date"] + pd.Timedelta(days=1)
            feat: dict = {"team_id": team_id, "as_of_date": as_of}
            if prior.empty:
                rows.append(feat)
                continue
            for w in windows:
                tail = prior.tail(w)
                if "win" in tail.columns:
                    feat[f"{prefix}win_pct_l{w}"] = tail["win"].mean()
                for src, dst_base in stat_map.items():
                    if src in tail.columns:
                        feat[f"{dst_base}_l{w}"] = tail[src].mean()
            rows.append(feat)

    return pd.DataFrame(rows)
