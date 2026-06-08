"""Ingest per-game team stats via nba_api (traditional + optional advanced)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoreadvancedv2, teamgamelog
from nba_api.stats.static import teams as nba_teams

from ml.ingest.nba_util import REQUEST_DELAY_S, parse_matchup, season_to_nba_id, throttle


def _team_index() -> dict[int, str]:
    return {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}


def fetch_team_game_log(team_id: int, season: str) -> pd.DataFrame:
    season_id = season_to_nba_id(season)
    log = teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season_id,
        season_type_all_star="Regular Season",
    )
    df = log.get_data_frames()[0]
    if df.empty:
        return df

    abbr_map = _team_index()
    df = df.copy()
    df.rename(
        columns={
            "Team_ID": "team_id",
            "GAME_ID": "game_id",
            "Game_ID": "game_id",
            "GAME_DATE": "game_date",
            "MATCHUP": "matchup",
            "WL": "wl",
            "PTS": "pts",
            "FG_PCT": "fg_pct",
            "FG3_PCT": "fg3_pct",
            "FT_PCT": "ft_pct",
            "REB": "reb",
            "AST": "ast",
            "STL": "stl",
            "BLK": "blk",
            "TOV": "tov",
            "PF": "pf",
        },
        inplace=True,
    )
    # Normalize column casing from nba_api variants
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)

    if "team_id" not in df.columns:
        df["team_id"] = team_id

    df["team_abbr"] = df["team_id"].map(abbr_map).fillna("UNK")
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed", utc=False)
    df["season"] = season
    df["win"] = (df["wl"] == "W").astype(int)

    is_home_list: list[bool] = []
    opp_list: list[str] = []
    for m in df["matchup"]:
        is_home, opp = parse_matchup(str(m))
        is_home_list.append(is_home)
        opp_list.append(opp)
    df["is_home"] = is_home_list
    df["opponent_abbr"] = opp_list
    df["source"] = "nba_api"
    df["ingested_at"] = pd.Timestamp.utcnow()
    return df


def ingest_team_game_logs(
    seasons: list[str],
    team_ids: list[int] | None = None,
    delay_s: float = REQUEST_DELAY_S,
) -> pd.DataFrame:
    ids = team_ids or [t["id"] for t in nba_teams.get_teams()]
    abbr_map = _team_index()
    last_at = [0.0]
    frames: list[pd.DataFrame] = []

    for season in seasons:
        for team_id in ids:
            abbr = abbr_map.get(team_id, str(team_id))
            print(f"nba_api team log: {abbr} {season}...")
            throttle(last_at, delay_s)
            try:
                df = fetch_team_game_log(team_id, season)
            except Exception as exc:
                print(f"  Warning: failed {abbr} {season}: {exc}")
                continue
            if df.empty:
                print(f"  Warning: no rows for {abbr} {season}")
                continue
            print(f"  {len(df)} games")
            frames.append(df)

    if not frames:
        raise RuntimeError("No team game logs fetched.")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["game_id", "team_id"]).sort_values(
        ["season", "team_id", "game_date"]
    )
    return out.reset_index(drop=True)


def fetch_game_advanced(game_id: str) -> pd.DataFrame:
    adv = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
    df = adv.get_data_frames()[0]
    if df.empty:
        return df
    df = df.copy()
    df.rename(
        columns={
            "GAME_ID": "game_id",
            "TEAM_ID": "team_id",
            "TEAM_ABBREVIATION": "team_abbr",
            "OFF_RATING": "off_rating",
            "DEF_RATING": "def_rating",
            "NET_RATING": "net_rating",
            "PACE": "pace",
            "PIE": "pie",
            "TS_PCT": "ts_pct",
            "EFG_PCT": "efg_pct",
        },
        inplace=True,
    )
    cols = {c: c.lower() for c in df.columns}
    return df.rename(columns=cols)


def ingest_game_advanced_stats(
    game_ids: list[str],
    delay_s: float = REQUEST_DELAY_S,
) -> pd.DataFrame:
    last_at = [0.0]
    frames: list[pd.DataFrame] = []
    for i, game_id in enumerate(game_ids):
        if i % 50 == 0:
            print(f"nba_api advanced: {i}/{len(game_ids)}...")
        throttle(last_at, delay_s)
        try:
            df = fetch_game_advanced(game_id)
        except Exception as exc:
            print(f"  Warning: {game_id}: {exc}")
            continue
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["game_id", "team_id"])


def merge_logs_with_advanced(logs: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    if advanced.empty:
        return logs
    adv_cols = [
        c
        for c in (
            "game_id",
            "team_id",
            "off_rating",
            "def_rating",
            "net_rating",
            "pace",
            "ts_pct",
            "efg_pct",
            "pie",
        )
        if c in advanced.columns
    ]
    return logs.merge(advanced[adv_cols], on=["game_id", "team_id"], how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA team game logs via nba_api")
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=["2022-23", "2023-24", "2024-25"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/team_game_logs.parquet"),
    )
    parser.add_argument(
        "--with-advanced",
        action="store_true",
        help="Fetch advanced box scores per game (slow; ~1 req/game)",
    )
    parser.add_argument(
        "--games",
        type=Path,
        default=Path("data/games.parquet"),
        help="Games file used to enumerate game_ids for --with-advanced",
    )
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_S)
    parser.add_argument(
        "--teams",
        nargs="*",
        default=None,
        help="Team abbreviations e.g. LAL BOS (default: all 30)",
    )
    args = parser.parse_args()

    team_ids = None
    if args.teams:
        abbr_to_id = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}
        team_ids = [abbr_to_id[a.upper()] for a in args.teams]

    logs = ingest_team_game_logs(args.seasons, team_ids=team_ids, delay_s=args.delay)

    if args.with_advanced:
        if not args.games.exists():
            raise FileNotFoundError(f"Games file required for --with-advanced: {args.games}")
        games = pd.read_parquet(args.games)
        if args.seasons:
            games = games[games["season"].isin(args.seasons)]
        game_ids = games["game_id"].astype(str).unique().tolist()
        print(f"Fetching advanced stats for {len(game_ids)} games...")
        advanced = ingest_game_advanced_stats(game_ids, delay_s=args.delay)
        logs = merge_logs_with_advanced(logs, advanced)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    logs.to_parquet(args.output, index=False)
    print(f"Wrote {len(logs)} rows -> {args.output}")


if __name__ == "__main__":
    main()
