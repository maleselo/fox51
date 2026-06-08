"""Fetch NBA game results via nba_api and persist to parquet."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# Standard delay to avoid rate limits on stats.nba.com
_REQUEST_DELAY_S = 0.6


def _seasons_to_years(season: str) -> tuple[int, int]:
    """Convert '2023-24' -> (2023, 2024)."""
    start, end = season.split("-")
    return int(start), int(f"20{end}")


def fetch_season_games(season: str) -> pd.DataFrame:
    """Return regular-season game rows for one NBA season string."""
    year_start, year_end = _seasons_to_years(season)
    season_id = f"{year_start}-{str(year_end)[-2:]}"

    # Regular season only (exclude preseason/playoffs in v0)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season_id,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    raw = finder.get_data_frames()[0]
    time.sleep(_REQUEST_DELAY_S)

    if raw.empty:
        return pd.DataFrame()

    # Two rows per game (one per team). Keep home team row for pairing.
    raw = raw.copy()
    raw["is_home"] = raw["MATCHUP"].str.contains(" vs. ", regex=False)

    home = raw[raw["is_home"]].rename(
        columns={
            "GAME_ID": "game_id",
            "GAME_DATE": "game_date",
            "TEAM_ID": "home_team_id",
            "TEAM_ABBREVIATION": "home_team",
            "PTS": "home_pts",
            "WL": "home_wl",
        }
    )[["game_id", "game_date", "home_team_id", "home_team", "home_pts", "home_wl"]]

    away = raw[~raw["is_home"]].rename(
        columns={
            "GAME_ID": "game_id",
            "TEAM_ID": "away_team_id",
            "TEAM_ABBREVIATION": "away_team",
            "PTS": "away_pts",
            "WL": "away_wl",
        }
    )[["game_id", "away_team_id", "away_team", "away_pts", "away_wl"]]

    games = home.merge(away, on="game_id", how="inner")
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = season
    games["home_win"] = (games["home_wl"] == "W").astype(int)
    games["margin"] = games["home_pts"] - games["away_pts"]
    games["total_pts"] = games["home_pts"] + games["away_pts"]
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return games


def ingest_seasons(seasons: list[str], output_path: Path) -> pd.DataFrame:
    frames = []
    for season in seasons:
        print(f"Fetching season {season}...")
        df = fetch_season_games(season)
        if df.empty:
            print(f"  Warning: no games for {season}")
            continue
        print(f"  {len(df)} games")
        frames.append(df)

    if not frames:
        raise RuntimeError("No games fetched. Check season strings and network.")

    all_games = pd.concat(frames, ignore_index=True)
    all_games = all_games.drop_duplicates(subset=["game_id"]).sort_values(
        ["game_date", "game_id"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_games.to_parquet(output_path, index=False)
    print(f"Wrote {len(all_games)} games -> {output_path}")
    return all_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA game results")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
        help="Season strings e.g. 2023-24",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/games.parquet"),
    )
    args = parser.parse_args()
    ingest_seasons(args.seasons, args.output)


if __name__ == "__main__":
    main()
