"""Generate performance visualizations from a backtest run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from ml.backtest.metrics import (
    brier_score,
    calibration_buckets,
    log_loss,
    max_drawdown_pct,
)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_calibration(preds: pd.DataFrame, n_bins: int = 10) -> plt.Figure:
    y = preds["home_win"].to_numpy()
    p = preds["home_win_prob"].to_numpy()
    buckets = calibration_buckets(y, p, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    valid = buckets["count"] > 0
    ax.scatter(
        buckets.loc[valid, "avg_pred"],
        buckets.loc[valid, "actual_rate"],
        s=buckets.loc[valid, "count"] * 2,
        alpha=0.75,
        label="Bins (size = count)",
        zorder=3,
    )
    ax.plot(
        buckets.loc[valid, "avg_pred"],
        buckets.loc[valid, "actual_rate"],
        color="#2563eb",
        alpha=0.6,
        zorder=2,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Actual home win rate")
    ax.set_title("Calibration")
    ax.legend(loc="lower right", fontsize=8)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_rolling_metrics(preds: pd.DataFrame, window: int = 100) -> plt.Figure:
    df = preds.sort_values("game_date").copy()
    df["correct"] = ((df["home_win_prob"] >= 0.5) == df["home_win"]).astype(float)
    df["brier_game"] = (df["home_win_prob"] - df["home_win"]) ** 2

    df["rolling_accuracy"] = df["correct"].rolling(window, min_periods=20).mean()
    df["rolling_brier"] = df["brier_game"].rolling(window, min_periods=20).mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(df["game_date"], df["rolling_accuracy"], color="#16a34a", linewidth=1.5)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel(f"Accuracy ({window}-game rolling)")
    ax1.set_ylim(0.35, 0.75)
    ax1.set_title("Performance over time")
    _style_axes(ax1)

    ax2.plot(df["game_date"], df["rolling_brier"], color="#dc2626", linewidth=1.5)
    ax2.set_ylabel(f"Brier ({window}-game rolling)")
    ax2.set_xlabel("Date")
    _style_axes(ax2)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_prediction_distribution(preds: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    wins = preds[preds["home_win"] == 1]["home_win_prob"]
    losses = preds[preds["home_win"] == 0]["home_win_prob"]

    bins = np.linspace(0, 1, 21)
    ax.hist(losses, bins=bins, alpha=0.55, label="Away won", color="#ef4444")
    ax.hist(wins, bins=bins, alpha=0.55, label="Home won", color="#22c55e")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel("Predicted P(home win)")
    ax.set_ylabel("Games")
    ax.set_title("Prediction distribution by outcome")
    ax.legend()
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_season_breakdown(preds: pd.DataFrame) -> plt.Figure:
    if "season" not in preds.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No season column", ha="center", va="center")
        ax.axis("off")
        return fig

    rows = []
    for season, g in preds.groupby("season"):
        y = g["home_win"].to_numpy()
        p = g["home_win_prob"].to_numpy()
        rows.append(
            {
                "season": season,
                "games": len(g),
                "brier": brier_score(y, p),
                "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
                "home_win_rate": float(y.mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("season")

    x = np.arange(len(summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, summary["brier"], width, label="Brier (lower is better)", color="#f97316")
    ax.bar(x + width / 2, summary["accuracy"], width, label="Accuracy @ 50%", color="#3b82f6")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["season"])
    ax.set_ylabel("Score")
    ax.set_title("Metrics by season")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _style_axes(ax)
    fig.tight_layout()
    return fig


def plot_betting_equity(bets: pd.DataFrame, initial_bankroll: float) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})

    bets = bets.sort_values("game_date").copy()
    equity = initial_bankroll + bets["pnl"].cumsum()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    ax1.plot(bets["game_date"], equity, color="#2563eb", linewidth=1.5)
    ax1.axhline(initial_bankroll, color="gray", linestyle=":", linewidth=1)
    ax1.set_ylabel("Bankroll (€)")
    ax1.set_title("Simulated betting performance")
    _style_axes(ax1)

    ax2.fill_between(bets["game_date"], drawdown, 0, color="#dc2626", alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    _style_axes(ax2)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_summary_dashboard(
    preds: pd.DataFrame,
    metrics: dict,
    bets: pd.DataFrame | None,
    initial_bankroll: float,
) -> plt.Figure:
    y = preds["home_win"].to_numpy()
    p = preds["home_win_prob"].to_numpy()
    buckets = calibration_buckets(y, p, n_bins=10)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("fox51 — Model performance report", fontsize=14, fontweight="bold", y=0.98)

    # 1. Calibration
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1)
    valid = buckets["count"] > 0
    ax1.scatter(
        buckets.loc[valid, "avg_pred"],
        buckets.loc[valid, "actual_rate"],
        s=buckets.loc[valid, "count"] * 1.5,
        c="#2563eb",
        alpha=0.8,
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Calibration")
    _style_axes(ax1)

    # 2. Rolling accuracy
    ax2 = fig.add_subplot(2, 2, 2)
    df = preds.sort_values("game_date")
    correct = ((df["home_win_prob"] >= 0.5) == df["home_win"]).astype(float)
    rolling = correct.rolling(100, min_periods=20).mean()
    ax2.plot(df["game_date"], rolling, color="#16a34a", linewidth=1.2)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax2.set_title("Rolling accuracy (100 games)")
    ax2.set_ylim(0.35, 0.75)
    _style_axes(ax2)

    # 3. Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    bins = np.linspace(0, 1, 21)
    ax3.hist(
        preds[preds["home_win"] == 0]["home_win_prob"],
        bins=bins,
        alpha=0.55,
        label="Away won",
        color="#ef4444",
    )
    ax3.hist(
        preds[preds["home_win"] == 1]["home_win_prob"],
        bins=bins,
        alpha=0.55,
        label="Home won",
        color="#22c55e",
    )
    ax3.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax3.set_xlabel("P(home win)")
    ax3.legend(fontsize=8)
    ax3.set_title("Prediction distribution")
    _style_axes(ax3)

    # 4. Summary text + optional equity
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    pred_m = metrics.get("predictions", {})
    lines = [
        "Summary",
        "─" * 28,
        f"Games evaluated : {pred_m.get('n_games', len(preds)):,}",
        f"Brier score     : {pred_m.get('brier', brier_score(y, p)):.4f}",
        f"Log loss        : {pred_m.get('log_loss', log_loss(y, p)):.4f}",
        f"Accuracy @ 50%  : {pred_m.get('accuracy_at_50', 0):.1%}",
        f"Pred home rate  : {pred_m.get('mean_predicted_home_win', p.mean()):.1%}",
        f"Actual home rate: {pred_m.get('actual_home_win_rate', y.mean()):.1%}",
    ]
    if bets is not None and not bets.empty:
        bet_m = metrics.get("betting", {})
        lines.extend(
            [
                "",
                "Betting simulation",
                "─" * 28,
                f"Bets placed     : {bet_m.get('n_bets', len(bets)):,}",
                f"ROI             : {bet_m.get('roi', 0):.1%}",
                f"Hit rate        : {bet_m.get('hit_rate', 0):.1%}",
                f"Max drawdown    : {bet_m.get('max_drawdown_pct', 0):.1%}",
                f"Final bankroll  : €{bet_m.get('final_bankroll', initial_bankroll):,.2f}",
            ]
        )
    ax4.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax4.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _write_html_report(run_dir: Path, image_names: list[str], metrics: dict) -> None:
    pred = metrics.get("predictions", {})
    cards = f"""
    <div class="metrics">
      <div class="card"><span>Brier</span><strong>{pred.get('brier', 0):.4f}</strong></div>
      <div class="card"><span>Log loss</span><strong>{pred.get('log_loss', 0):.4f}</strong></div>
      <div class="card"><span>Accuracy</span><strong>{pred.get('accuracy_at_50', 0):.1%}</strong></div>
      <div class="card"><span>Games</span><strong>{pred.get('n_games', 0):,}</strong></div>
    </div>
    """
    imgs = "\n".join(
        f'<section><h2>{name.replace("_", " ").replace(".png", "").title()}</h2>'
        f'<img src="{name}" alt="{name}"/></section>'
        for name in image_names
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>fox51 backtest report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin-bottom: 0.25rem; }}
    .metrics {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
    .card {{ background: white; padding: 1rem 1.25rem; border-radius: 8px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); min-width: 120px; }}
    .card span {{ display: block; font-size: 0.8rem; color: #64748b; }}
    .card strong {{ font-size: 1.4rem; }}
    section {{ background: white; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;
               box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>fox51 — Model performance</h1>
  <p>Run: <code>{run_dir.name}</code></p>
  {cards}
  {imgs}
</body>
</html>
"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")


def generate_report(run_dir: Path, dpi: int = 120) -> Path:
    """Build all charts for a backtest run directory. Returns path to dashboard PNG."""
    run_dir = Path(run_dir)
    preds_path = run_dir / "predictions.parquet"
    metrics_path = run_dir / "metrics.json"

    if not preds_path.exists():
        raise FileNotFoundError(f"No predictions at {preds_path}")

    preds = pd.read_parquet(preds_path)
    metrics: dict = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    bets_path = run_dir / "bets.parquet"
    bets = pd.read_parquet(bets_path) if bets_path.exists() else None
    initial_bankroll = float(
        metrics.get("config", {}).get("betting", {}).get("initial_bankroll_eur", 1000.0)
    )

    charts_dir = run_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    figures: list[tuple[str, plt.Figure]] = [
        ("dashboard.png", plot_summary_dashboard(preds, metrics, bets, initial_bankroll)),
        ("calibration.png", plot_calibration(preds)),
        ("rolling_metrics.png", plot_rolling_metrics(preds)),
        ("prediction_distribution.png", plot_prediction_distribution(preds)),
        ("season_breakdown.png", plot_season_breakdown(preds)),
    ]
    if bets is not None and not bets.empty:
        figures.append(("betting_equity.png", plot_betting_equity(bets, initial_bankroll)))

    saved: list[str] = []
    for name, fig in figures:
        out = charts_dir / name
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(f"charts/{name}")

    _write_html_report(run_dir, saved, metrics)
    return charts_dir / "dashboard.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate backtest performance charts")
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Backtest run directory (default: latest in data/backtest_runs)",
    )
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    if args.run_dir is None:
        runs_root = Path("data/backtest_runs")
        candidates = sorted(runs_root.glob("*_elo_v0"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise SystemExit("No backtest runs found. Run fox51-backtest first.")
        run_dir = candidates[-1]
    else:
        run_dir = args.run_dir

    out = generate_report(run_dir, dpi=args.dpi)
    print(f"Report written -> {run_dir}/report.html")
    print(f"Dashboard      -> {out}")


if __name__ == "__main__":
    main()
