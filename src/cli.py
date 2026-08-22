"""
Command-line entry point for the EPL Forecaster.

Usage
-----
  python cli.py init                       – create / migrate the database
  python cli.py load-history               – download all historical seasons
  python cli.py load-history --from 2015   – only seasons from 2015/16 onwards
  python cli.py sync                       – sync current season fixtures/results
  python cli.py fetch-values               – fetch squad market values (Transfermarkt)
  python cli.py fetch-values --season 2026/27
  python cli.py pre-season                 – pre-season forecast (no matches needed)
  python cli.py pre-season --season 2026/27 --sims 10000
  python cli.py forecast                   – blended forecast (auto-detects matchday)
  python cli.py forecast --season 2026/27
  python cli.py standings --season 2025/26
  python cli.py calibrate                  – walk-forward CV to tune hyperparams
  python cli.py calibrate --seasons 2023/24 2024/25 2025/26 --sims 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db import get_connection, init_db
from fetch_historical import load_all_historical
from fetch_live import sync_current_season
from fetch_transfers import sync_squad_values
from model import InSeasonForecaster, PreSeasonForecaster, blend_alpha

# 2026/27 promoted teams (update each year)
PROMOTED_2026_27 = {
    "Coventry": "champions",
    "Ipswich":  "runners_up",
    "Hull":     "playoff",
}

# 2026/27 gross transfer spend (USD millions, money IN only) — used as fallback
# when Transfermarkt squad value data is not yet available for the new season.
# Gross spend (not net) reflects squad reinforcement without penalising clubs
# that sold a superstar. Source: ESPN / Squawka, as of 22 Aug 2026.
NET_SPEND_2026_27 = {
    "Chelsea":        439.4,
    "Arsenal":        270.6,
    "Newcastle":      224.1,
    "Man City":       204.3,
    "Ipswich":        199.6,
    "Aston Villa":    180.1,
    "Tottenham":      302.9,   # gross = £302m even if net is lower after sales
    "Liverpool":      118.1,
    "Man United":     118.4,
    "Brighton":       133.1,
    "Coventry":       153.6,
    "Fulham":         103.1,
    "Hull":           110.2,
    "Leeds":          107.4,
    "Nottingham":      45.1,
    "Everton":         77.6,
    "Bournemouth":     71.0,
    "Brentford":       89.8,
    "Sunderland":      31.7,
    "Crystal Palace":  42.2,
}


def _current_season() -> str:
    import datetime as dt
    today = dt.date.today()
    start = today.year if today.month >= 8 else today.year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def cmd_init(_args) -> None:
    init_db()


def cmd_load_history(args) -> None:
    from_year = getattr(args, "from_year", 1993)
    load_all_historical(from_year=from_year)


def cmd_sync(args) -> None:
    year = getattr(args, "year", None)
    sync_current_season(start_year=year)


def cmd_fetch_values(args) -> None:
    season = getattr(args, "season", None) or _current_season()
    sync_squad_values(season)


def cmd_pre_season(args) -> None:
    """Pure pre-season forecast – ignores any in-season results."""
    season = getattr(args, "season", None) or _current_season()
    sims   = getattr(args, "sims", 10_000)
    season_start = int(season.split("/")[0])
    promoted = PROMOTED_2026_27 if season_start == 2026 else {}
    if not promoted:
        print(f"Note: no promoted teams configured for {season}. Edit PROMOTED_2026_27 in cli.py.")
    print(f"Pre-season forecast for {season} …")
    f = PreSeasonForecaster(season, promoted_teams=promoted)
    f.fit()
    f.predict(n_simulations=sims).print_table()


def cmd_forecast(args) -> None:
    """
    Smart forecast: automatically blends pre-season prior with current-season
    data based on how many matchdays have been played.

    Before matchday 1: equivalent to pre-season (100% prior).
    After matchday 1:  prior fades quickly; at matchday 19 it's ~17%.
    At all times:      15% pedigree residual from historical form is retained.
    """
    season   = getattr(args, "season", None) or _current_season()
    sims     = getattr(args, "sims", 10_000)
    out_path = getattr(args, "output", "docs/data.json")

    season_start = int(season.split("/")[0])
    promoted = PROMOTED_2026_27 if season_start == 2026 else {}
    spend    = NET_SPEND_2026_27 if season_start == 2026 else None

    pre = PreSeasonForecaster(season, promoted_teams=promoted, spend_values=spend)
    pre.fit()

    f = InSeasonForecaster(season, prior_params=pre.team_params,
                           prior_home_adv=pre.home_advantage)
    f.fit()
    result = f.predict(n_simulations=sims)
    result.print_table()

    # Load current standings to enrich the JSON output
    with get_connection() as conn:
        standings = [dict(r) for r in conn.execute("""
            SELECT team, played, won, drawn, lost, gf, ga, gd, points
            FROM standings WHERE season = ?
            ORDER BY points DESC, gd DESC, gf DESC
        """, (season,)).fetchall()]

    result.write_json(path=out_path, standings=standings)


# Known promoted teams per eval season (used by calibrate command).
# Update each summer after promotion/relegation is confirmed.
PROMOTED_HISTORY = {
    "2022/23": {"Fulham": "champions", "Bournemouth": "runners_up", "Nottm Forest": "playoff"},
    "2023/24": {"Sheffield United": "champions", "Burnley": "runners_up", "Luton": "playoff"},
    "2024/25": {"Leicester": "champions", "Ipswich": "runners_up", "Southampton": "playoff"},
    "2025/26": {"Sunderland": "champions", "Leeds": "runners_up", "Coventry": "playoff"},
}


def cmd_calibrate(args) -> None:
    """
    Walk-forward cross-validation to tune blend hyperparameters.

    For each eval season, trains only on seasons strictly before it,
    then evaluates at multiple matchday checkpoints. Reports which
    (tau, alpha_max, xi_prior, xi_current) combination gives the best
    average Spearman rank-correlation, penalised for variance across seasons.
    """
    from calibrate import calibrate, print_calibration_results

    # Which seasons to evaluate against
    default_eval = ["2023/24", "2024/25", "2025/26"]
    eval_seasons = getattr(args, "seasons", None) or default_eval

    sims      = getattr(args, "sims", 2_000)
    top_n     = getattr(args, "top_n", 15)
    fast_mode = getattr(args, "fast", False)

    # In fast mode use a coarser grid for a quick sanity check
    if fast_mode:
        tau_grid        = [5.0, 10.0]
        alpha_max_grid  = [0.80, 0.85]
        xi_prior_grid   = [0.003]
        xi_current_grid = [0.010]
        checkpoints     = [5, 19, 38]
        sims            = min(sims, 500)
    else:
        tau_grid        = None   # use calibrate() defaults
        alpha_max_grid  = None
        xi_prior_grid   = None
        xi_current_grid = None
        checkpoints     = None

    results = calibrate(
        eval_seasons=eval_seasons,
        checkpoints=checkpoints,
        n_simulations=sims,
        tau_grid=tau_grid,
        alpha_max_grid=alpha_max_grid,
        xi_prior_grid=xi_prior_grid,
        xi_current_grid=xi_current_grid,
        promoted_by_season={s: PROMOTED_HISTORY.get(s, {}) for s in eval_seasons},
    )
    print_calibration_results(results, top_n=top_n)


def cmd_standings(args) -> None:
    season = getattr(args, "season", None) or _current_season()
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT team, played, won, drawn, lost, gf, ga, gd, points
            FROM standings
            WHERE season = ?
            ORDER BY points DESC, gd DESC, gf DESC
        """, (season,)).fetchall()

    if not rows:
        print(f"No data for season {season}. Run 'sync' or 'load-history' first.")
        return

    print(f"\n{'#':>3}  {'Team':<28} {'P':>3} {'W':>3} {'D':>3} {'L':>3} "
          f"{'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}")
    print("─" * 65)
    for i, r in enumerate(rows, start=1):
        print(f"{i:>3}  {r['team']:<28} {r['played']:>3} {r['won']:>3} "
              f"{r['drawn']:>3} {r['lost']:>3} {r['gf']:>4} {r['ga']:>4} "
              f"{r['gd']:>+4} {r['points']:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPL End-of-Season Rank Forecaster")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialise the database")

    p_hist = sub.add_parser("load-history", help="Load historical seasons")
    p_hist.add_argument("--from", dest="from_year", type=int, default=1993)

    p_sync = sub.add_parser("sync", help="Sync current season results/fixtures")
    p_sync.add_argument("--year", type=int, default=None)

    p_val = sub.add_parser("fetch-values", help="Fetch squad market values from Transfermarkt")
    p_val.add_argument("--season", default=None, help="e.g. 2026/27")

    p_pre = sub.add_parser("pre-season", help="Pre-season rank forecast (before first match)")
    p_pre.add_argument("--season", default=None, help="e.g. 2026/27")
    p_pre.add_argument("--sims", type=int, default=10_000)

    p_fc = sub.add_parser("forecast", help="Blended forecast — auto-detects matchday")
    p_fc.add_argument("--season", default=None)
    p_fc.add_argument("--sims", type=int, default=10_000)
    p_fc.add_argument("--output", default="docs/data.json",
                      help="Path to write JSON output (default: docs/data.json)")

    p_st = sub.add_parser("standings", help="Show current standings")
    p_st.add_argument("--season", default=None)

    p_cal = sub.add_parser("calibrate", help="Walk-forward CV to tune hyperparameters")
    p_cal.add_argument("--seasons", nargs="+", default=None,
                       metavar="SEASON", help="e.g. 2023/24 2024/25 2025/26")
    p_cal.add_argument("--sims", type=int, default=2_000,
                       help="Simulations per evaluation (default 2000; use 5000+ for final run)")
    p_cal.add_argument("--top-n", dest="top_n", type=int, default=15,
                       help="How many param combos to print")
    p_cal.add_argument("--fast", action="store_true",
                       help="Coarse grid + fewer checkpoints for a quick sanity check")

    args = parser.parse_args()

    dispatch = {
        "init":         cmd_init,
        "load-history": cmd_load_history,
        "sync":         cmd_sync,
        "fetch-values": cmd_fetch_values,
        "pre-season":   cmd_pre_season,
        "forecast":     cmd_forecast,
        "standings":    cmd_standings,
        "calibrate":    cmd_calibrate,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
