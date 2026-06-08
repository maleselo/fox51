"""Backtest evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def accuracy_at_50(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    preds = (y_prob >= 0.5).astype(int)
    return float(np.mean(preds == y_true))


def calibration_buckets(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.cut(df["p"], bins=n_bins, labels=False)
    grouped = df.groupby("bin", observed=True).agg(
        count=("y", "size"),
        avg_pred=("p", "mean"),
        actual_rate=("y", "mean"),
    )
    grouped["gap"] = grouped["actual_rate"] - grouped["avg_pred"]
    return grouped.reset_index()


def roi(bets: pd.DataFrame) -> float:
    if bets.empty or bets["stake"].sum() == 0:
        return 0.0
    return float(bets["pnl"].sum() / bets["stake"].sum())


def max_drawdown_pct(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(abs(dd.min()))


def summarize_predictions(preds: pd.DataFrame) -> dict:
    y = preds["home_win"].to_numpy()
    p = preds["home_win_prob"].to_numpy()
    return {
        "n_games": int(len(preds)),
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "accuracy_at_50": accuracy_at_50(y, p),
        "mean_predicted_home_win": float(p.mean()),
        "actual_home_win_rate": float(y.mean()),
    }


def summarize_bets(bets: pd.DataFrame, initial_bankroll: float) -> dict:
    if bets.empty:
        return {
            "n_bets": 0,
            "roi": 0.0,
            "total_pnl": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "final_bankroll": initial_bankroll,
        }

    equity = initial_bankroll + bets["pnl"].cumsum()
    return {
        "n_bets": int(len(bets)),
        "roi": roi(bets),
        "total_pnl": float(bets["pnl"].sum()),
        "hit_rate": float(bets["won"].mean()),
        "avg_edge": float(bets["edge"].mean()),
        "max_drawdown_pct": max_drawdown_pct(equity),
        "final_bankroll": float(equity.iloc[-1]),
    }


def write_run_report(
    run_dir: Path,
    config: dict,
    preds: pd.DataFrame,
    bets: pd.DataFrame | None,
    initial_bankroll: float,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(run_dir / "predictions.parquet", index=False)

    report: dict = {
        "config": config,
        "predictions": summarize_predictions(preds),
        "calibration": calibration_buckets(
            preds["home_win"].to_numpy(), preds["home_win_prob"].to_numpy()
        ).to_dict(orient="records"),
    }

    if bets is not None and not bets.empty:
        bets.to_parquet(run_dir / "bets.parquet", index=False)
        report["betting"] = summarize_bets(bets, initial_bankroll)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report
