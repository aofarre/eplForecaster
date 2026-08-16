"""
Walk-forward cross-validation and hyperparameter calibration.

Design principles
─────────────────
* NO temporal leakage: when evaluating season S, we only train on seasons
  strictly before S, and simulate the season from each checkpoint using
  only matches played up to that point.
* Robustness over fit: we optimise for *average* rank-correlation across
  multiple held-out seasons, not a single season.  A model that performs
  well on one season but poorly on others is worse than one that is
  consistently decent.
* Early-season checkpoints matter most: getting matchday-5 forecasts right
  is harder and arguably more useful than matchday-35 ones.  We weight
  checkpoints inversely by matchday (or equally, per user preference).

Evaluation metric
─────────────────
For each (season, checkpoint):
  * Run the blended forecast with the given hyperparams.
  * Compare predicted median rank vs actual final rank for all 20 teams.
  * Spearman rank-correlation: 1 = perfect, 0 = random, -1 = inverted.
  * MARE (Mean Absolute Rank Error): average |predicted_median − actual| rank.

Grid searched
─────────────
  tau          blend transition speed  (matchdays to half-saturation)
  alpha_max    max current-season weight (1 - alpha_max = pedigree floor)
  xi_prior     time-decay for prior-season fitting
  xi_current   time-decay for current-season fitting

history_seasons is fixed at 2 (tuning it separately via nested CV is
tractable but adds significant compute; 2 seasons is a well-motivated
choice that avoids stale data from relegated/promoted teams).
"""

from __future__ import annotations

import itertools
import math
from typing import NamedTuple

import numpy as np

from db import get_connection
from model import (
    HOME_ADVANTAGE_INIT,
    InSeasonForecaster,
    PreSeasonForecaster,
    TeamParams,
    _fit_dc_mle,
    _simulate,
    _time_weights,
    blend_alpha,
)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank-correlation between two equal-length lists."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def _rankdata(values: list[float]) -> list[float]:
    """Rank values (1 = smallest). Handles ties with average rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _mare(predicted: list[float], actual: list[float]) -> float:
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


# ── Actual final standings ────────────────────────────────────────────────────

def _get_actual_final_ranks(season: str) -> dict[str, int]:
    """
    Compute actual final ranks from the DB for a completed season.
    Returns {team: rank} where 1 = champion.
    """
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT team, points, gd, gf
            FROM standings
            WHERE season = ?
            ORDER BY points DESC, gd DESC, gf DESC
        """, (season,)).fetchall()
    return {r["team"]: i + 1 for i, r in enumerate(rows)}


# ── Walk-forward evaluator ────────────────────────────────────────────────────

class CheckpointResult(NamedTuple):
    season: str
    matchday: int
    n_played: int
    spearman: float
    mare: float


def evaluate_season(
    eval_season: str,
    checkpoints: list[int],        # matchdays to evaluate at, e.g. [5,10,19,30,38]
    tau: float,
    alpha_max: float,
    xi_prior: float,
    xi_current: float,
    history_seasons: int = 2,
    n_simulations: int = 2_000,    # fewer for calibration speed
    prior_params: dict[str, TeamParams] | None = None,
    prior_home_adv: float = HOME_ADVANTAGE_INIT,
) -> list[CheckpointResult]:
    """
    Evaluate one held-out season at multiple matchday checkpoints.

    If prior_params is provided (pre-computed) it is reused, avoiding
    re-running the expensive MLE for every (tau, alpha_max) combo.
    """
    actual_ranks = _get_actual_final_ranks(eval_season)
    if not actual_ranks:
        return []

    # Load all matches for eval season, sorted by date
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT date, home_team, away_team, home_goals, away_goals, result, matchday
            FROM matches WHERE season = ? ORDER BY date
        """, (eval_season,)).fetchall()

    all_matches = [dict(r) for r in rows]
    played_all  = [m for m in all_matches if m["result"] is not None]

    # Build {matchday: list[match]} index
    def matchday_of(m: dict) -> int:
        if m.get("matchday"):
            return int(m["matchday"])
        # Infer from position in played list (10 games per round)
        return (played_all.index(m) // 10) + 1 if m in played_all else 999

    results = []

    for cp in checkpoints:
        # Matches played up to and including this checkpoint
        played_up_to = [m for m in played_all if matchday_of(m) <= cp]
        remaining    = [m for m in all_matches if m not in played_up_to
                        and m["result"] is not None]
        # "remaining" for simulation = everything not yet played at checkpoint
        sim_remaining = [m for m in all_matches if matchday_of(m) > cp]

        if not played_up_to and cp > 0:
            continue

        # Blend parameters
        actual_matchday = matchday_of(played_up_to[-1]) if played_up_to else 0
        alpha = blend_alpha(actual_matchday, tau=tau, alpha_max=alpha_max)

        if not played_up_to:
            blended = dict(prior_params) if prior_params else {}
            home_adv = prior_home_adv
        else:
            teams = sorted(
                {m["home_team"] for m in all_matches} | {m["away_team"] for m in all_matches}
            )
            weights = _time_weights(played_up_to, xi_current)
            curr_params, curr_home_adv, _ = _fit_dc_mle(
                played_up_to, weights, teams, initial_params=prior_params
            )
            blended = {}
            for t in teams:
                pr = (prior_params or {}).get(t, TeamParams(t))
                cu = curr_params.get(t, TeamParams(t))
                blended[t] = TeamParams(
                    t,
                    attack  = (1 - alpha) * pr.attack  + alpha * cu.attack,
                    defence = (1 - alpha) * pr.defence + alpha * cu.defence,
                )
            home_adv = (1 - alpha) * prior_home_adv + alpha * curr_home_adv

        if not blended:
            continue

        team_list, ranks_arr = _simulate(
            blended, played_up_to, sim_remaining, home_adv, n_simulations
        )
        median_ranks = {t: float(np.median(ranks_arr[:, i]))
                        for i, t in enumerate(team_list)}

        common = [t for t in actual_ranks if t in median_ranks]
        if len(common) < 10:
            continue

        pred   = [median_ranks[t] for t in common]
        actual = [actual_ranks[t] for t in common]

        results.append(CheckpointResult(
            season=eval_season,
            matchday=cp,
            n_played=len(played_up_to),
            spearman=_spearman(pred, actual),
            mare=_mare(pred, actual),
        ))

    return results


# ── Grid search ──────────────────────────────────────────────────────────────

class GridResult(NamedTuple):
    tau: float
    alpha_max: float
    xi_prior: float
    xi_current: float
    mean_spearman: float
    std_spearman: float
    mean_mare: float
    n_obs: int


def calibrate(
    eval_seasons: list[str],
    checkpoints: list[int] | None = None,
    history_seasons: int = 2,
    n_simulations: int = 2_000,
    tau_grid:        list[float] | None = None,
    alpha_max_grid:  list[float] | None = None,
    xi_prior_grid:   list[float] | None = None,
    xi_current_grid: list[float] | None = None,
    promoted_by_season: dict[str, dict[str, str]] | None = None,
) -> list[GridResult]:
    """
    Walk-forward cross-validation over `eval_seasons`.

    For each eval season:
      1. Fit a pre-season prior using only seasons strictly before it
         (reused across all (tau, alpha_max) combos for speed).
      2. Sweep the hyperparameter grid.
      3. Evaluate at each checkpoint.

    Returns GridResult list sorted by mean Spearman (best first).
    """
    if checkpoints is None:
        checkpoints = [1, 3, 5, 10, 19, 30, 38]

    tau_grid        = tau_grid        or [3.0, 5.0, 8.0, 12.0, 20.0]
    alpha_max_grid  = alpha_max_grid  or [0.75, 0.80, 0.85, 0.90]
    xi_prior_grid   = xi_prior_grid   or [0.002, 0.003, 0.005, 0.008]
    xi_current_grid = xi_current_grid or [0.005, 0.010, 0.020]
    promoted_by_season = promoted_by_season or {}

    n_combos = (len(tau_grid) * len(alpha_max_grid) *
                len(xi_prior_grid) * len(xi_current_grid))
    print(f"Calibration: {len(eval_seasons)} seasons × {len(checkpoints)} checkpoints "
          f"× {n_combos} param combos = "
          f"~{len(eval_seasons)*len(checkpoints)*n_combos:,} evaluations")
    print(f"(using {n_simulations:,} sims per evaluation)\n")

    # Outer loop: eval seasons
    # For each (eval_season, xi_prior) pair, fit the prior once and cache it.
    # Then sweep (tau, alpha_max, xi_current) cheaply.
    all_obs: dict[tuple, list[tuple[float, float]]] = {}  # (tau,amax,xip,xic) → [(spear,mare)]

    for eval_season in eval_seasons:
        print(f"── Evaluating held-out season: {eval_season} ──")

        for xi_prior in xi_prior_grid:
            print(f"  Fitting prior (xi_prior={xi_prior}) …", end=" ", flush=True)
            promoted = promoted_by_season.get(eval_season, {})
            pre = PreSeasonForecaster(
                eval_season,
                promoted_teams=promoted,
                history_seasons=history_seasons,
                time_decay_xi=xi_prior,
            )
            pre.fit()
            print("done")

            for tau, alpha_max, xi_current in itertools.product(
                tau_grid, alpha_max_grid, xi_current_grid
            ):
                key = (tau, alpha_max, xi_prior, xi_current)
                cp_results = evaluate_season(
                    eval_season=eval_season,
                    checkpoints=checkpoints,
                    tau=tau,
                    alpha_max=alpha_max,
                    xi_prior=xi_prior,
                    xi_current=xi_current,
                    history_seasons=history_seasons,
                    n_simulations=n_simulations,
                    prior_params=pre.team_params,
                    prior_home_adv=pre.home_advantage,
                )
                for r in cp_results:
                    all_obs.setdefault(key, []).append((r.spearman, r.mare))

        print()

    # Aggregate and sort
    grid_results = []
    for (tau, alpha_max, xi_prior, xi_current), obs in all_obs.items():
        spearmans = [o[0] for o in obs if not math.isnan(o[0])]
        mares     = [o[1] for o in obs]
        if not spearmans:
            continue
        grid_results.append(GridResult(
            tau=tau,
            alpha_max=alpha_max,
            xi_prior=xi_prior,
            xi_current=xi_current,
            mean_spearman=float(np.mean(spearmans)),
            std_spearman=float(np.std(spearmans)),
            mean_mare=float(np.mean(mares)),
            n_obs=len(spearmans),
        ))

    grid_results.sort(key=lambda r: -r.mean_spearman)
    return grid_results


def print_calibration_results(results: list[GridResult], top_n: int = 20) -> None:
    print(f"\n{'tau':>5} {'α_max':>6} {'ξ_prior':>8} {'ξ_curr':>7}"
          f" {'Spear↑':>7} {'±std':>5} {'MARE↓':>6} {'N':>5}")
    print("─" * 57)
    for r in results[:top_n]:
        print(f"{r.tau:>5.0f} {r.alpha_max:>6.2f} {r.xi_prior:>8.3f} {r.xi_current:>7.3f}"
              f" {r.mean_spearman:>7.3f} {r.std_spearman:>5.3f}"
              f" {r.mean_mare:>6.2f} {r.n_obs:>5}")

    best = results[0]
    print(f"\nBest params:  tau={best.tau}  alpha_max={best.alpha_max}"
          f"  xi_prior={best.xi_prior}  xi_current={best.xi_current}")
    print(f"  Spearman: {best.mean_spearman:.3f} ± {best.std_spearman:.3f}"
          f"  MARE: {best.mean_mare:.2f} ranks")
    print("\nNote: prefer params with low std_spearman (robust across seasons)")
    print("      over those with high mean but also high std (overfit to specific seasons).")


if __name__ == "__main__":
    # Quick demo: evaluate last 3 completed seasons
    EVAL_SEASONS = ["2023/24", "2024/25", "2025/26"]

    # Known promoted teams for each eval season
    PROMOTED = {
        "2023/24": {"Luton": "playoff", "Sheffield United": "champions", "Burnley": "runners_up"},
        "2024/25": {"Leicester": "champions", "Ipswich": "runners_up", "Southampton": "playoff"},
        "2025/26": {"Sunderland": "champions", "Leeds": "runners_up", "Coventry": "playoff"},
    }

    results = calibrate(
        eval_seasons=EVAL_SEASONS,
        checkpoints=[1, 5, 10, 19, 38],
        n_simulations=1_000,          # use more (5000+) for final calibration
        tau_grid=[3.0, 5.0, 10.0],
        alpha_max_grid=[0.80, 0.85, 0.90],
        xi_prior_grid=[0.003, 0.005],
        xi_current_grid=[0.010, 0.020],
        promoted_by_season=PROMOTED,
    )
    print_calibration_results(results)
