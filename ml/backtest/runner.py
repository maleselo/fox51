"""Walk-forward backtest runner (v0: Elo moneyline)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from ml.betting.devig import devig_two_way, edge as compute_edge
from ml.betting.kelly import StakeConfig, settle_bet, suggested_stake_eur
from ml.features.build import build_game_features
from ml.ingest.odds import load_odds
from ml.models.elo import EloModel


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_moneyline_odds(
    odds: pd.DataFrame, game_id: str, book: str, game_date: pd.Timestamp
) -> tuple[float, float] | None:
    """
    Return (home_decimal, away_decimal) for the latest capture before tip-off.
    v0: captured_at must be < game_date (date-level); upgrade to tip time later.
    """
    g = odds[
        (odds["game_id"] == game_id)
        & (odds["book"] == book)
        & (odds["market"] == "moneyline")
    ].copy()
    if g.empty:
        return None

    g = g[g["captured_at"] < pd.Timestamp(game_date, tz="UTC")]
    if g.empty:
        return None

    latest = g.sort_values("captured_at").groupby("selection").tail(1)
    home_row = latest[latest["selection"] == "home"]
    away_row = latest[latest["selection"] == "away"]
    if home_row.empty or away_row.empty:
        return None

    return float(home_row["decimal_odds"].iloc[0]), float(away_row["decimal_odds"].iloc[0])


def run_backtest(config: dict) -> dict:
    games_path = Path(config["data"]["games_path"])
    if not games_path.exists():
        raise FileNotFoundError(
            f"Games file not found: {games_path}. Run: fox51-ingest"
        )

    games = build_game_features(pd.read_parquet(games_path))
    seasons = config.get("seasons")
    if seasons:
        games = games[games["season"].isin(seasons)].reset_index(drop=True)

    wf = config["walk_forward"]
    min_train = int(wf["min_train_games"])

    elo_cfg = config["model"]["elo"]
    model = EloModel(
        k_factor=float(elo_cfg["k_factor"]),
        home_court_elo=float(elo_cfg["home_court_elo"]),
        initial_rating=float(elo_cfg["initial_rating"]),
    )

    pred_rows: list[dict] = []
    bet_rows: list[dict] = []

    odds_path = Path(config["data"]["odds_path"])
    odds = load_odds(odds_path)
    has_odds = odds is not None and not odds.empty

    bet_cfg = config.get("betting", {})
    stake_config = StakeConfig(
        kelly_fraction=float(bet_cfg.get("kelly_fraction", 0.25)),
        max_bet_pct=float(bet_cfg.get("max_bet_pct", 0.05)),
        min_stake_eur=float(bet_cfg.get("min_stake_eur", 2.0)),
        min_edge=float(bet_cfg.get("min_edge", 0.03)),
    )
    books: list[str] = list(bet_cfg.get("french_books", []))
    bankroll = float(bet_cfg.get("initial_bankroll_eur", 1000.0))
    initial_bankroll = bankroll

    for i, row in enumerate(games.itertuples(index=False)):
        if i < min_train:
            model.update(row.home_team_id, row.away_team_id, int(row.home_win))
            continue

        p_home = model.predict_home_win_prob(row.home_team_id, row.away_team_id)
        pred_rows.append(
            {
                "game_id": row.game_id,
                "game_date": row.game_date,
                "season": row.season,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "home_win": int(row.home_win),
                "home_win_prob": p_home,
                "home_rest_days": row.home_rest_days,
                "away_rest_days": row.away_rest_days,
                "rest_diff": row.rest_diff,
            }
        )

        if has_odds:
            for book in books:
                pair = _select_moneyline_odds(odds, row.game_id, book, row.game_date)
                if pair is None:
                    continue
                home_odds, away_odds = pair
                fair_home, fair_away = devig_two_way(home_odds, away_odds)

                for side, model_p, fair_p, dec_odds, won in (
                    ("home", p_home, fair_home, home_odds, bool(row.home_win)),
                    ("away", 1 - p_home, fair_away, away_odds, not bool(row.home_win)),
                ):
                    e = compute_edge(model_p, fair_p)
                    stake = suggested_stake_eur(
                        model_p, dec_odds, bankroll, e, stake_config
                    )
                    if stake <= 0:
                        continue
                    pnl = settle_bet(stake, dec_odds, won)
                    bankroll += pnl
                    bet_rows.append(
                        {
                            "game_id": row.game_id,
                            "game_date": row.game_date,
                            "book": book,
                            "selection": side,
                            "decimal_odds": dec_odds,
                            "model_prob": model_p,
                            "fair_implied_prob": fair_p,
                            "edge": e,
                            "stake": stake,
                            "won": int(won),
                            "pnl": pnl,
                            "bankroll_after": bankroll,
                        }
                    )

        model.update(row.home_team_id, row.away_team_id, int(row.home_win))

    preds = pd.DataFrame(pred_rows)
    bets = pd.DataFrame(bet_rows) if bet_rows else pd.DataFrame()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output"]["dir"]) / f"{run_id}_elo_v0"
    from ml.backtest.metrics import write_run_report

    report = write_run_report(run_dir, config, preds, bets if has_odds else None, initial_bankroll)
    report["run_dir"] = str(run_dir)
    report["odds_used"] = has_odds

    from ml.backtest.report import generate_report

    dashboard = generate_report(run_dir)
    report["report_html"] = str(run_dir / "report.html")
    report["dashboard_png"] = str(dashboard)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run walk-forward NBA backtest")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/backtest.yaml"),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_backtest(config)
    print(json.dumps({k: v for k, v in report.items() if k != "calibration"}, indent=2))
    print(f"\nMetrics  -> {report['run_dir']}/metrics.json")
    print(f"Charts   -> {report.get('report_html', 'N/A')}")


if __name__ == "__main__":
    main()
