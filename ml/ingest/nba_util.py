"""Shared helpers for nba_api ingest."""

from __future__ import annotations

REQUEST_DELAY_S = 0.6


def season_to_nba_id(season: str) -> str:
    """Convert '2023-24' -> '2023-24' NBA season id."""
    start, end = season.split("-")
    return f"{int(start)}-{end}"


def throttle(last_at: list[float], delay_s: float = REQUEST_DELAY_S) -> None:
    """Sleep if needed to respect rate limits. Pass mutable one-item list for state."""
    import time as _time

    elapsed = _time.monotonic() - last_at[0]
    if last_at[0] and elapsed < delay_s:
        _time.sleep(delay_s - elapsed)
    last_at[0] = _time.monotonic()


def parse_matchup(matchup: str) -> tuple[bool, str]:
    """
    Parse MATCHUP like 'LAL @ DEN' or 'LAL vs. DEN'.
    Returns (is_home, opponent_abbr).
    """
    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        return True, parts[1].strip()
    if " @ " in matchup:
        parts = matchup.split(" @ ")
        return False, parts[1].strip()
    return False, ""
