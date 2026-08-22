"""
EPL End-of-Season Rank Forecaster
==================================

Three forecasting modes, automatically selected
────────────────────────────────────────────────

1. PRE-SEASON  (PreSeasonForecaster)
   Before the first match is played.  Uses:
   * Dixon-Coles parameters fitted on the prior 2 seasons (time-decayed MLE)
   * Transfermarkt squad value adjustment
   * Promoted team baselines derived from historical promoted-team data

2. IN-SEASON  (InSeasonForecaster)
   Once matches have started.  Blends the pre-season prior with
   a freshly fitted current-season estimate via the pedigree blend function.

3. BLEND FUNCTION  (blend_alpha)
   Controls how quickly the prior is phased out.

   alpha(n) = alpha_max * (1 - exp(-n / tau))

   where n = average matchdays played per team.
   Default:  tau = 5,  alpha_max = 0.85

   | Matchday | Prior weight | Current weight | Interpretation               |
   |----------+--------------+----------------+------------------------------|
   |     0    |    100 %     |      0 %       | pure pre-season prior        |
   |     1    |     85 %     |     15 %       | one game -- prior still leads |
   |     5    |     57 %     |     43 %       | early-season transition      |
   |    10    |     34 %     |     66 %       | current season takes over    |
   |    19    |     17 %     |     83 %       | mid-season                   |
   |    38    |    ~15 %     |    ~85 %       | end of season                |

   The permanent 15 % floor is the PEDIGREE COMPONENT.  It encodes the
   idea that a club's history of winning (or losing) is weakly informative
   even when we have a full current-season picture.  Arsenal or Man City
   reaching 80+ points in the prior two seasons will always carry a small
   forward-looking signal even at matchday 38.

Dixon-Coles MLE
───────────────
Parameters:  attack[i], defence[i] for every team, home_advantage, rho.
Likelihood:  product of per-match Poisson probabilities with a low-score
             correction factor tau for (0,0), (1,0), (0,1), (1,1) scorelines.
Time-decay:  each match is weighted by exp(-xi * days_ago).
Fitting:     scipy.optimize.minimize (L-BFGS-B).  Falls back to the
             iterative Poisson regression if scipy is unavailable.

References
──────────
  Dixon & Coles (1997) "Modelling Association Football Scores and
  Inefficiencies in the Football Betting Market", Applied Statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from db import get_connection

# -- Constants ----------------------------------------------------------------

HOME_ADVANTAGE_INIT = 0.25

BLEND_TAU       = 20.0  # matchdays for half-transition (calibrated via walk-forward CV)
BLEND_ALPHA_MAX = 0.80  # max current-season weight; pedigree floor = 20% (calibrated)
XI_PRIOR        = 0.002  # time-decay for prior-season MLE fitting (calibrated)
XI_CURRENT      = 0.010  # time-decay for current-season MLE fitting (calibrated)

TRANSFER_K = 0.10

# Promoted team attack/defence offsets from bottom-5 EPL average (log-scale).
# Historically, no promoted team has finished top-4 in the modern CL era (since 2001).
# Typical outcomes: champions ~14th, runners-up ~15th, playoff ~17th.
# These offsets must be negative enough to reflect that gulf vs established EPL sides.
PROMOTED_OFFSET = {
    "champions":  -0.35,   # championship winners: typically finish ~14th
    "runners_up": -0.45,   # runners-up: typically ~15-16th
    "playoff":    -0.55,   # playoff winners: typically ~16-18th
}
# After transfer adjustment, cap promoted teams so they can't exceed the
# attack strength of the 8th-best returning team (i.e. they can't be projected
# top-half purely from squad value data which may be stale/overrated).
PROMOTED_ATTACK_CAP_RANK = 8  # cannot exceed the Nth-best returning team's attack

# -- Blend weight -------------------------------------------------------------

def blend_alpha(matchdays_played: float, tau: float = BLEND_TAU,
                alpha_max: float = BLEND_ALPHA_MAX) -> float:
    """
    Weight placed on current-season estimates vs the pre-season prior.

    Returns alpha in [0, alpha_max].  The prior always contributes 1 - alpha,
    preserving historical pedigree throughout the season.
    """
    return alpha_max * (1.0 - math.exp(-matchdays_played / tau))


# -- Data structures ----------------------------------------------------------

@dataclass
class TeamParams:
    name: str
    attack: float = 0.0    # log-scale; 0 = league average
    defence: float = 0.0   # log-scale; higher = worse defence


@dataclass
class ForecastResult:
    season: str
    teams: list
    matchdays_played: float = 0.0
    blend_alpha_used: float = 0.0
    simulated_ranks: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def summary(self):
        rows = []
        for i, team in enumerate(self.teams):
            ranks = self.simulated_ranks[:, i]
            rows.append({
                "team":         team,
                "median_rank":  float(np.median(ranks)),
                "p10_rank":     float(np.percentile(ranks, 10)),
                "p90_rank":     float(np.percentile(ranks, 90)),
                "p_title":      float(np.mean(ranks == 1)),
                "p_top4":       float(np.mean(ranks <= 4)),
                "p_relegation": float(np.mean(ranks >= 18)),
            })
        return sorted(rows, key=lambda r: r["median_rank"])

    def print_table(self) -> None:
        if self.matchdays_played > 0:
            prior_pct = (1 - self.blend_alpha_used) * 100
            curr_pct  = self.blend_alpha_used * 100
            print(f"\n  Season {self.season}  |  Matchday ~{self.matchdays_played:.0f}  |"
                  f"  Blend: {curr_pct:.0f}% current season + {prior_pct:.0f}% prior (pedigree)")
        else:
            print(f"\n  Season {self.season}  |  Pre-season forecast  |  100% prior")

        print(f"\n{'Team':<28} {'Med':>4} {'P10':>4} {'P90':>4}"
              f" {'Title%':>7} {'Top4%':>6} {'Rel%':>5}")
        print("-" * 63)
        for r in self.summary():
            print(f"{r['team']:<28} {r['median_rank']:>4.0f}"
                  f" {r['p10_rank']:>4.0f} {r['p90_rank']:>4.0f}"
                  f" {r['p_title']*100:>6.1f}%"
                  f" {r['p_top4']*100:>5.1f}%"
                  f" {r['p_relegation']*100:>4.1f}%")

    def to_json(self, standings: list[dict] | None = None) -> dict:
        """
        Produce the JSON payload written to docs/data.json and consumed by the
        static site.  `standings` is a list of current-season table rows (from
        the DB `standings` view); if omitted, actual stats are left as nulls.
        """
        import datetime as dt

        # Build lookup: team → current standings row
        st = {r["team"]: r for r in (standings or [])}

        if self.matchdays_played > 0:
            blend_label = (f"Matchday {self.matchdays_played:.0f} — "
                           f"{self.blend_alpha_used*100:.0f}% current season + "
                           f"{(1-self.blend_alpha_used)*100:.0f}% prior")
        else:
            blend_label = "Pre-season forecast — 100% prior"

        rows = []
        for r in self.summary():
            t = r["team"]
            s = st.get(t, {})
            team_idx = self.teams.index(t)
            rows.append({
                "team":         t,
                # Current standings (null if season hasn't started)
                "played":       s.get("played"),
                "won":          s.get("won"),
                "drawn":        s.get("drawn"),
                "lost":         s.get("lost"),
                "gf":           s.get("gf"),
                "ga":           s.get("ga"),
                "gd":           s.get("gd"),
                "points":       s.get("points"),
                # Forecast (proj_rank assigned after dedup below)
                "_mean_rank":   float(np.mean(self.simulated_ranks[:, team_idx])),
                "_points":      s.get("points") or 0,
                "rank_p10":     round(r["p10_rank"]),
                "rank_p90":     round(r["p90_rank"]),
                "p_title":      round(r["p_title"] * 100, 1),
                "p_top4":       round(r["p_top4"] * 100, 1),
                "p_top6":       round(float(
                    np.mean(self.simulated_ranks[:, team_idx] <= 6)
                ) * 100, 1),
                "p_relegation": round(r["p_relegation"] * 100, 1),
            })

        # Assign unique proj_rank 1–N with no ties.
        # Primary sort: mean simulated rank (continuous, rarely identical).
        # Tiebreaker: current points desc, then gd desc, then alphabetical.
        rows.sort(key=lambda r: (
            r["_mean_rank"],
            -(r["_points"]),
            -(r.get("gd") or 0),
            r["team"],
        ))
        for pos, r in enumerate(rows, 1):
            r["proj_rank"] = pos
        # Clean up internal sort keys
        for r in rows:
            del r["_mean_rank"], r["_points"]

        return {
            "season":       self.season,
            "matchday":     self.matchdays_played,
            "blend_label":  blend_label,
            "updated_at":   dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "teams":        rows,
        }

    def write_json(self, path: str = "docs/data.json",
                   standings: list[dict] | None = None) -> None:
        import json, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = self.to_json(standings)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote forecast → {path}")


# -- Dixon-Coles helpers ------------------------------------------------------

def _dc_tau(h, a, mu_h, mu_a, rho):
    if h == 0 and a == 0: return 1.0 - mu_h * mu_a * rho
    if h == 1 and a == 0: return 1.0 + mu_a * rho
    if h == 0 and a == 1: return 1.0 + mu_h * rho
    if h == 1 and a == 1: return 1.0 - rho
    return 1.0

def _poisson_log_pmf(k, mu):
    if mu <= 0 or k < 0: return -1e9
    return k * math.log(mu) - mu - math.lgamma(k + 1)

def _match_log_lik(h, a, mu_h, mu_a, rho):
    tau = _dc_tau(h, a, mu_h, mu_a, rho)
    if tau <= 1e-10: return -1e9
    return math.log(tau) + _poisson_log_pmf(h, mu_h) + _poisson_log_pmf(a, mu_a)


# -- Dixon-Coles MLE ----------------------------------------------------------

def _fit_dc_mle(matches, weights, teams, initial_params=None):
    """
    Fit attack/defence via Dixon-Coles MLE (scipy L-BFGS-B).
    Falls back to iterative Poisson if scipy is unavailable.
    Returns (team_params dict, home_advantage, rho).
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        print("  [model] scipy not found -- using iterative Poisson fallback.")
        params, home_adv = _fit_iterative(matches, weights, teams, initial_params)
        return params, home_adv, 0.0

    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # Parameter vector has 2n+1 elements:
    #   [attack_1..N-1 (n-1), defence_0..N-1 (n), home_adv (1), rho (1)]
    # attack[0] is the reference team, fixed at 0 for identifiability.
    def unpack(x):
        atk = np.zeros(n); atk[1:] = x[:n-1]
        def_ = x[n-1:2*n-1]
        return atk, def_, x[-2], x[-1]

    x0 = np.zeros(2 * n + 1)
    if initial_params:
        for i, t in enumerate(teams[1:]):
            if t in initial_params: x0[i] = initial_params[t].attack
        for i, t in enumerate(teams):
            if t in initial_params: x0[n-1+i] = initial_params[t].defence
    x0[-2] = HOME_ADVANTAGE_INIT
    x0[-1] = -0.1

    def neg_ll(x):
        atk, def_, home_adv, rho = unpack(x)
        if not (-0.99 < rho < 0.99): return 1e9
        total = 0.0
        for i, m in enumerate(matches):
            hi = team_idx.get(m["home_team"])
            ai = team_idx.get(m["away_team"])
            if hi is None or ai is None: continue
            mu_h = math.exp(float(atk[hi]) - float(def_[ai]) + home_adv)
            mu_a = math.exp(float(atk[ai]) - float(def_[hi]))
            total += weights[i] * _match_log_lik(int(m["home_goals"]), int(m["away_goals"]), mu_h, mu_a, rho)
        return -total

    bounds = [(-3, 3)] * (n-1) + [(-3, 3)] * n + [(0.0, 0.6)] + [(-0.5, 0.0)]
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-8})

    atk, def_, home_adv, rho = unpack(res.x)
    mean_atk = float(np.mean(atk))
    atk -= mean_atk; def_ -= mean_atk

    return (
        {t: TeamParams(t, float(atk[i]), float(def_[i])) for i, t in enumerate(teams)},
        float(home_adv),
        float(rho),
    )


def _fit_iterative(matches, weights, teams, initial_params=None, n_iters=20):
    """Iterative Poisson regression (no scipy needed). Converges in ~15 passes."""
    attack  = {t: (initial_params[t].attack  if initial_params and t in initial_params else 0.0) for t in teams}
    defence = {t: (initial_params[t].defence if initial_params and t in initial_params else 0.0) for t in teams}
    home_adv = HOME_ADVANTAGE_INIT

    for _ in range(n_iters):
        for t in teams:
            num = den = 0.0
            for i, m in enumerate(matches):
                w = weights[i]
                if m["home_team"] == t:
                    mu = math.exp(attack[t] - defence[m["away_team"]] + home_adv)
                    num += w * m["home_goals"]; den += w * mu
                elif m["away_team"] == t:
                    mu = math.exp(attack[t] - defence[m["home_team"]])
                    num += w * m["away_goals"]; den += w * mu
            if den > 0: attack[t] = math.log(max(num/den, 0.01)) + attack[t]

        for t in teams:
            num = den = 0.0
            for i, m in enumerate(matches):
                w = weights[i]
                if m["home_team"] == t:
                    mu = math.exp(attack[m["away_team"]] - defence[t])
                    num += w * m["away_goals"]; den += w * mu
                elif m["away_team"] == t:
                    mu = math.exp(attack[m["home_team"]] - defence[t] + home_adv)
                    num += w * m["home_goals"]; den += w * mu
            if den > 0: defence[t] = math.log(max(num/den, 0.01)) + defence[t]

        num = sum(weights[i] * m["home_goals"] for i, m in enumerate(matches))
        den = sum(weights[i] * math.exp(attack[m["home_team"]] - defence[m["away_team"]] + home_adv)
                  for i, m in enumerate(matches))
        if den > 0: home_adv = math.log(max(num/den, 0.01)) + home_adv

        mean_atk = sum(attack.values()) / len(attack)
        attack  = {t: v - mean_atk for t, v in attack.items()}
        defence = {t: v - mean_atk for t, v in defence.items()}

    return {t: TeamParams(t, attack[t], defence[t]) for t in teams}, home_adv


# -- Time-decay weights -------------------------------------------------------

def _time_weights(matches, xi):
    import datetime as dt
    today = dt.date.today()
    out = []
    for m in matches:
        try:
            days_ago = max((today - dt.date.fromisoformat(m.get("date", ""))).days, 0)
        except (ValueError, TypeError):
            days_ago = 0
        out.append(math.exp(-xi * days_ago))
    return out


# -- Monte Carlo simulation ---------------------------------------------------

def _simulate(team_params, played_matches, remaining_matches,
              home_advantage=HOME_ADVANTAGE_INIT, n_simulations=10_000):
    """Simulate remaining fixtures; return (team_list, ranks_array)."""
    teams = sorted(team_params.keys())
    ti = {t: i for i, t in enumerate(teams)}
    n  = len(teams)

    base_pts = np.zeros(n, int); base_gd = np.zeros(n, int); base_gf = np.zeros(n, int)
    for m in played_matches:
        hi = ti.get(m["home_team"]); ai = ti.get(m["away_team"])
        if hi is None or ai is None: continue
        hg, ag = m["home_goals"], m["away_goals"]
        base_gd[hi] += hg - ag; base_gd[ai] += ag - hg
        base_gf[hi] += hg;      base_gf[ai] += ag
        if   m["result"] == "H": base_pts[hi] += 3
        elif m["result"] == "A": base_pts[ai] += 3
        else:                    base_pts[hi] += 1; base_pts[ai] += 1

    fixtures = []
    for m in remaining_matches:
        h, a = m["home_team"], m["away_team"]
        if h not in ti or a not in ti: continue
        hp, ap = team_params[h], team_params[a]
        mu_h = math.exp(hp.attack - ap.defence + home_advantage)
        mu_a = math.exp(ap.attack - hp.defence)
        fixtures.append((ti[h], ti[a], mu_h, mu_a))

    rng = np.random.default_rng()
    all_ranks = np.empty((n_simulations, n), int)

    for sim in range(n_simulations):
        pts = base_pts.copy(); gd = base_gd.copy(); gf = base_gf.copy()
        for hi, ai, mu_h, mu_a in fixtures:
            hg = int(rng.poisson(mu_h)); ag = int(rng.poisson(mu_a))
            gd[hi] += hg-ag; gd[ai] += ag-hg
            gf[hi] += hg;    gf[ai] += ag
            if   hg > ag: pts[hi] += 3
            elif ag > hg: pts[ai] += 3
            else:         pts[hi] += 1; pts[ai] += 1
        order = sorted(range(n), key=lambda i: (-pts[i], -gd[i], -gf[i]))
        ranks = np.empty(n, int)
        for rank, idx in enumerate(order, 1): ranks[idx] = rank
        all_ranks[sim] = ranks

    return teams, all_ranks


# -- Pre-season forecaster ----------------------------------------------------

class PreSeasonForecaster:
    """
    Forecast before any matches are played, using prior seasons + transfers.

    Fit outputs are stored in `team_params` and `home_advantage`; pass these
    directly to InSeasonForecaster.prior_params once the season starts.
    """

    def __init__(self, season, promoted_teams=None, history_seasons=2,
                 time_decay_xi=XI_PRIOR):
        self.season = season
        self.promoted_teams = promoted_teams or {}
        self.history_seasons = history_seasons
        self.xi = time_decay_xi
        self.team_params = {}
        self.home_advantage = HOME_ADVANTAGE_INIT
        self.rho = -0.1

    def _prior_season_labels(self):
        start = int(self.season.split("/")[0])
        return [f"{start-i}/{str(start-i+1)[-2:]}" for i in range(1, self.history_seasons+1)]

    def _load_history(self):
        labels = self._prior_season_labels()
        ph = ",".join("?"*len(labels))
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT date,home_team,away_team,home_goals,away_goals FROM matches"
                f" WHERE season IN ({ph}) AND result IS NOT NULL ORDER BY date", labels
            ).fetchall()
        return [dict(r) for r in rows]

    def fit(self):
        matches = self._load_history()
        print(f"[pre-season] {len(matches)} matches from {self._prior_season_labels()}")

        all_teams = ({m["home_team"] for m in matches} | {m["away_team"] for m in matches})
        returning = sorted(all_teams - set(self.promoted_teams))
        ret_matches = [m for m in matches if m["home_team"] in returning and m["away_team"] in returning]
        ret_weights  = _time_weights(ret_matches, self.xi)

        self.team_params, self.home_advantage, self.rho = _fit_dc_mle(
            ret_matches, ret_weights, returning)
        print(f"  home_advantage={self.home_advantage:.3f}  rho={self.rho:.3f}")

        # Promoted team baselines
        bottom5 = sorted(self.team_params.values(), key=lambda p: p.attack)[:5]
        base_atk = sum(p.attack  for p in bottom5) / 5
        base_def = sum(p.defence for p in bottom5) / 5
        for team, route in self.promoted_teams.items():
            offset = PROMOTED_OFFSET.get(route, PROMOTED_OFFSET["playoff"])
            self.team_params[team] = TeamParams(team, base_atk+offset, base_def+offset)

        # Transfer value adjustment
        try:
            from fetch_transfers import get_squad_values
            values = get_squad_values(self.season)
            if values:
                avg = sum(values.values()) / len(values)
                for team, p in self.team_params.items():
                    if team in values and avg > 0:
                        adj = TRANSFER_K * math.log(values[team] / avg)
                        p.attack += adj; p.defence += adj
                print(f"  Transfer adjustment applied for {len(values)} clubs.")
            else:
                print("  [transfers] No squad values -- run: python src/cli.py fetch-values")
        except Exception as e:
            print(f"  [transfers] Skipped: {e}")

        # Cap promoted teams: they cannot be rated stronger than the Nth-best
        # returning side.  This prevents a single big win or stale transfer data
        # from pushing a newly promoted club into a top-4 projection.
        returning_attacks = sorted(
            [p.attack for t, p in self.team_params.items() if t not in self.promoted_teams],
            reverse=True
        )
        if len(returning_attacks) >= PROMOTED_ATTACK_CAP_RANK:
            cap_attack = returning_attacks[PROMOTED_ATTACK_CAP_RANK - 1]
            for team in self.promoted_teams:
                if team in self.team_params:
                    p = self.team_params[team]
                    if p.attack > cap_attack:
                        print(f"  [cap] {team}: attack capped {p.attack:+.3f} → {cap_attack:+.3f}")
                        p.attack = cap_attack

        print("[pre-season] Strength estimates:")
        for p in sorted(self.team_params.values(), key=lambda x: -x.attack):
            print(f"  {p.name:<28} atk={p.attack:+.3f}  def={p.defence:+.3f}")
        return self

    def predict(self, n_simulations=10_000):
        if not self.team_params: self.fit()
        all_t = sorted(self.team_params)
        remaining = [{"home_team": h, "away_team": a} for h in all_t for a in all_t if h != a]
        teams, ranks = _simulate(self.team_params, [], remaining,
                                  self.home_advantage, n_simulations)
        return ForecastResult(self.season, teams, 0, 0, ranks)


# -- In-season (blended) forecaster -------------------------------------------

class InSeasonForecaster:
    """
    Blends the pre-season prior with current-season Dixon-Coles estimates.

    alpha(n) = alpha_max * (1 - exp(-n / tau))

    The permanent (1 - alpha_max) residual is the pedigree component:
    historical strength always contributes a small amount, capturing
    the tendency of strong clubs to sustain performance over time.

    Parameters
    ----------
    season : str
        e.g. "2026/27"
    prior_params : dict[str, TeamParams]
        From PreSeasonForecaster.team_params
    prior_home_adv : float
        From PreSeasonForecaster.home_advantage
    blend_tau : float
        Transition speed (default 5 matchdays).
    blend_alpha_max : float
        Maximum current-season weight (default 0.85; 15% pedigree floor).
    time_decay_xi : float
        Within-season recency decay (default 0.01; recent matches matter more).
    """

    def __init__(self, season, prior_params, prior_home_adv=HOME_ADVANTAGE_INIT,
                 blend_tau=BLEND_TAU, blend_alpha_max=BLEND_ALPHA_MAX,
                 time_decay_xi=XI_CURRENT):
        self.season = season
        self.prior_params = prior_params
        self.prior_home_adv = prior_home_adv
        self.blend_tau = blend_tau
        self.blend_alpha_max = blend_alpha_max
        self.xi = time_decay_xi
        self.blended_params = {}
        self.blended_home_adv = prior_home_adv
        self.matchdays_played = 0.0
        self.alpha = 0.0
        self._played = []
        self._remaining = []

    def _load_data(self):
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT date,home_team,away_team,home_goals,away_goals,result,matchday
                FROM matches WHERE season=? ORDER BY date
            """, (self.season,)).fetchall()
        for r in rows:
            d = dict(r)
            (self._played if d["result"] else self._remaining).append(d)

    def _avg_matchdays(self):
        if not self._played: return 0.0
        mds = {m.get("matchday") for m in self._played} - {None}
        return float(max(mds)) if mds else len(self._played) / 10.0

    def fit(self):
        self._load_data()
        self.matchdays_played = self._avg_matchdays()
        self.alpha = blend_alpha(self.matchdays_played, self.blend_tau, self.blend_alpha_max)

        print(f"[in-season] {self.season} -- matchday ~{self.matchdays_played:.0f}"
              f" | alpha={self.alpha:.2f}"
              f" (current {self.alpha*100:.0f}%  prior/pedigree {(1-self.alpha)*100:.0f}%)")

        # Determine current-season teams from fixtures (played + remaining)
        current_teams = sorted(
            {m["home_team"] for m in self._played + self._remaining} |
            {m["away_team"] for m in self._played + self._remaining}
        )

        if not self._played:
            self.blended_params = {t: self.prior_params.get(t, TeamParams(t))
                                   for t in current_teams}
            return self

        # Don't attempt in-season MLE until we have at least half a full matchday
        # set — with fewer data points the MLE is too noisy and can push
        # promoted/lucky-start teams to absurd projected positions.
        min_matches_for_mle = max(len(current_teams) // 2, 5)
        if len(self._played) < min_matches_for_mle:
            print(f"  [in-season] Only {len(self._played)} matches — using prior only (need {min_matches_for_mle})")
            self.blended_params = {t: self.prior_params.get(t, TeamParams(t))
                                   for t in current_teams}
            return self

        weights = _time_weights(self._played, self.xi)
        print(f"  Fitting current-season DC-MLE on {len(self._played)} matches ...")
        curr_params, curr_home_adv, _ = _fit_dc_mle(
            self._played, weights, current_teams, initial_params=self.prior_params)

        a = self.alpha
        for team in current_teams:
            pr = self.prior_params.get(team, TeamParams(team))
            cu = curr_params.get(team, TeamParams(team))
            self.blended_params[team] = TeamParams(
                team,
                attack  = (1-a)*pr.attack  + a*cu.attack,
                defence = (1-a)*pr.defence + a*cu.defence,
            )
        self.blended_home_adv = (1-a)*self.prior_home_adv + a*curr_home_adv
        return self

    def predict(self, n_simulations=10_000):
        if not self.blended_params: self.fit()
        teams, ranks = _simulate(self.blended_params, self._played, self._remaining,
                                  self.blended_home_adv, n_simulations)
        return ForecastResult(self.season, teams, self.matchdays_played, self.alpha, ranks)


# Alias for backward compatibility
Forecaster = InSeasonForecaster


# -- CLI convenience ----------------------------------------------------------

if __name__ == "__main__":
    import sys
    season = sys.argv[1] if len(sys.argv) > 1 else "2026/27"
    promoted = {"Coventry": "champions", "Ipswich": "runners_up", "Hull": "playoff"}

    pre = PreSeasonForecaster(season, promoted_teams=promoted)
    pre.fit()

    f = InSeasonForecaster(season, prior_params=pre.team_params,
                           prior_home_adv=pre.home_advantage)
    f.fit()
    f.predict(n_simulations=1_000).print_table()
