# fox51

NBA value-bet research toolkit (v0): ingest game data, Elo baseline model, walk-forward backtest, and Kelly stake sizing for French bookmakers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -e .
```

## v0 workflow

### 1. Ingest NBA games

```bash
fox51-ingest --seasons 2022-23 2023-24 2024-25 --output data/games.parquet
```

Offline smoke test (no API):

```bash
python scripts/generate_sample_data.py --output data/games.parquet
```

### 2. (Optional) Add French book odds

Place a parquet file at `data/odds.parquet` with columns:

`game_id`, `book`, `market`, `selection`, `decimal_odds`, `line`, `captured_at`

Books: `winamax`, `betclic`, `unibet_fr`, `parions_sport`

Without odds, the backtest still runs **model evaluation** (Brier, log loss, calibration). Value-bet simulation requires odds.

### 3. Run backtest

```bash
fox51-backtest --config config/backtest.yaml
```

Outputs land in `data/backtest_runs/<timestamp>_elo_v0/`:

- `predictions.parquet` — per-game home win probability
- `bets.parquet` — simulated bets (if odds present)
- `metrics.json` — summary metrics
- `report.html` + `charts/` — performance visualizations (auto-generated)

Regenerate charts for an existing run:

```bash
fox51-report data/backtest_runs/<run_id>_elo_v0
# or latest run:
fox51-report
```

Open `report.html` in a browser to view calibration, rolling accuracy, and season breakdown.

### 4. Iterate

Copy `config/backtest.yaml` to `config/runs/my_experiment.yaml`, tweak Elo params or `min_edge`, re-run, and compare `metrics.json` across runs.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Project layout

```
ml/
  ingest/     NBA games + odds schema
  features/   Point-in-time features
  models/     Elo (v0); XGBoost later
  betting/    Devig + Kelly
  backtest/   Walk-forward runner + metrics
config/       Backtest YAML
data/         Parquet + run artifacts (gitignored)
```
