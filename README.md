# EPL Forecaster

End-of-season rank forecaster for the English Premier League, with a live GitHub Pages table.

🔗 **[View the forecast →](https://aofarre.github.io/eplForecaster/)**

## How it works

The model combines two signals:

1. **Dixon-Coles Poisson regression** fitted on historical seasons — estimates each team's attack and defence strength via maximum likelihood, with time-decay weighting so recent matches count more.

2. **Pre-season prior** — before matches are played, team strength is estimated from the prior 2 seasons plus Transfermarkt squad market values. Promoted teams get a baseline derived from historical promoted-team performance.

These are blended with a **pedigree function** that smoothly transitions from 100% prior at the start of the season to ~85% current-season data by the end. The permanent ~15% prior residual captures the historical tendency of strong clubs to sustain performance over multiple seasons.

```
blend α(n) = 0.85 × (1 − e^{−n/5})
```

| Matchday | Prior weight | Interpretation |
|---|---|---|
| 0 | 100% | Pure pre-season forecast |
| 1 | 85% | One game — prior still leads |
| 5 | 57% | Early transition |
| 10 | 34% | Current season takes over |
| 38 | ~15% | Pedigree residual only |

Monte Carlo simulation (10,000 runs) produces a full probability distribution over final standings.

## Usage

```bash
pip install -r requirements.txt

# One-time setup
python src/cli.py init
python src/cli.py load-history       # downloads 1993/94–present from football-data.co.uk

# Each matchday (or let GitHub Actions do it)
python src/cli.py sync               # fetch latest results
python src/cli.py fetch-values       # squad market values from Transfermarkt
python src/cli.py forecast           # run model, write docs/data.json

# Calibrate hyperparameters using walk-forward cross-validation
python src/cli.py calibrate --seasons 2023/24 2024/25 2025/26

# Show current standings
python src/cli.py standings
```

Set `FOOTBALL_DATA_API_KEY` env var for richer fixture data (free tier at [football-data.org](https://www.football-data.org/)).

## Refreshing the live site

The forecast refreshes automatically at **09:30, 12:00, 14:30, and 17:30 EST**
during the EPL season (one hour earlier during daylight saving time, because GitHub
Actions schedules use UTC).

To refresh on demand, open **[Run the update workflow](https://github.com/aofarre/eplForecaster/actions/workflows/update.yml)**, select **Run workflow**, and click **Run workflow** again. The site updates when that run completes. One or two manual refreshes per day, in addition to the scheduled runs, remains well within the score API limit.

### Weekly projection history

The projection dropdown defaults to the current forecast. Choose an earlier Monday
to see that week's forecast and standings as they were captured. History begins
when this feature is deployed; earlier forecasts cannot be reconstructed from
the current data.

Every Monday, year-round, the workflow triggers at both 22:00 and 23:00 UTC.
A timezone-aware guard runs the forecast only for the trigger corresponding to
**6:00 pm America/New_York** (22:00 UTC during daylight saving time, 23:00 UTC
otherwise). The other trigger stops before downloading data or running the model.
The eligible run syncs results, generates a fresh forecast, and saves it in
`docs/snapshots/YYYY-MM-DD.json` with an entry in `docs/snapshots/index.json`.
The scheduled time is shown in the selector; the actual capture and forecast
update times are shown when viewing a snapshot. GitHub Actions can start late,
and data fetching and simulation take time, so capture is not instantaneous.
If the eligible trigger starts after midnight New York time, is dropped, or fails,
that week's snapshot is missed rather than being backdated. Existing snapshots
are never overwritten by a rerun; manual refreshes update only the current forecast.
Archive files and the index are committed with the live forecast and served by
GitHub Pages.

## Data sources

- Match results: [football-data.co.uk](https://www.football-data.co.uk) (historical) + [football-data.org](https://www.football-data.org) / [OpenFootball](https://github.com/openfootball/football.json) (live)
- Squad market values: [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)

## References

Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Applied Statistics*, 46(2), 265–280.
