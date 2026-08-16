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

## Data sources

- Match results: [football-data.co.uk](https://www.football-data.co.uk) (historical) + [football-data.org](https://www.football-data.org) / [OpenFootball](https://github.com/openfootball/football.json) (live)
- Squad market values: [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)

## References

Dixon, M. J. & Coles, S. G. (1997). Modelling Association Football Scores and Inefficiencies in the Football Betting Market. *Applied Statistics*, 46(2), 265–280.
