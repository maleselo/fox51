"""Odds parquet schema and import helpers for French bookmakers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ODDS_COLUMNS = [
    "game_id",
    "book",  # winamax | betclic | unibet_fr | parions_sport
    "market",  # moneyline | spread | total
    "selection",  # home | away | over | under
    "decimal_odds",
    "line",  # handicap or total line; NaN for moneyline
    "captured_at",  # ISO timestamp — must be before tip-off for honest backtests
]

FRENCH_BOOKS = frozenset({"winamax", "betclic", "unibet_fr", "parions_sport"})


def validate_odds(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(ODDS_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Odds parquet missing columns: {sorted(missing)}")

    out = df.copy()
    out["captured_at"] = pd.to_datetime(out["captured_at"], utc=True)
    out["decimal_odds"] = out["decimal_odds"].astype(float)

    unknown_books = set(out["book"].unique()) - FRENCH_BOOKS
    if unknown_books:
        raise ValueError(f"Unknown book(s): {unknown_books}. Expected one of {FRENCH_BOOKS}")

    if (out["decimal_odds"] <= 1.0).any():
        raise ValueError("decimal_odds must be > 1.0")

    return out


def load_odds(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return validate_odds(pd.read_parquet(path))


def write_template(path: Path) -> None:
    """Write an empty odds template for manual / scripted fills."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=ODDS_COLUMNS).to_parquet(path, index=False)
    print(f"Wrote empty odds template -> {path}")
