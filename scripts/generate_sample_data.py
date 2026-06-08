"""Generate synthetic NBA games for offline backtest smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_games(n_teams: int = 30, games_per_team: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    team_ids = list(range(1, n_teams + 1))
    team_abbr = {i: f"T{i:02d}" for i in team_ids}

    rows: list[dict] = []
    game_idx = 0
    for _ in range(games_per_team * n_teams // 2):
        home, away = rng.choice(team_ids, size=2, replace=False)
        home_strength = rng.normal(0, 1)
        home_win = int(rng.random() < 1 / (1 + np.exp(-(home_strength + 0.15))))
        margin = rng.integers(1, 15) * (1 if home_win else -1)
        home_pts = int(rng.integers(95, 125))
        away_pts = home_pts - margin
        game_idx += 1
        rows.append(
            {
                "game_id": f"SAMPLE{game_idx:05d}",
                "game_date": pd.Timestamp("2023-10-15") + pd.Timedelta(days=game_idx // 8),
                "season": "2023-24",
                "home_team_id": int(home),
                "home_team": team_abbr[home],
                "away_team_id": int(away),
                "away_team": team_abbr[away],
                "home_pts": home_pts,
                "away_pts": away_pts,
                "home_wl": "W" if home_win else "L",
                "away_wl": "L" if home_win else "W",
                "home_win": home_win,
                "margin": margin,
                "total_pts": home_pts + away_pts,
            }
        )

    return pd.DataFrame(rows).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/games.parquet"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_games()
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} sample games -> {args.output}")


if __name__ == "__main__":
    main()
