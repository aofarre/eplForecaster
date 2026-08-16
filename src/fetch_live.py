"""
Fetch current-season EPL results and fixtures.

Two backends are supported (tried in order):

1. football-data.org API  – requires a free API key (10 req/min on free tier)
   Register at: https://www.football-data.org/client/register
   Set env var:  FOOTBALL_DATA_API_KEY=<your_key>

2. OpenFootball JSON (GitHub raw) – no key required, updated daily.
   URL: https://raw.githubusercontent.com/openfootball/football.json/master/{YY}-{ZZ}/en.1.json
"""

import os
from datetime import datetime

import requests

from db import get_connection, init_db, upsert_match

# ── football-data.org ────────────────────────────────────────────────────────

FDORG_BASE = "https://api.football-data.org/v4"
FDORG_PL_CODE = "PL"


def _fdorg_headers() -> dict:
    key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    return {"X-Auth-Token": key} if key else {}


def _fdorg_result(match: dict) -> str | None:
    score = match.get("score", {})
    full = score.get("fullTime", {})
    home, away = full.get("home"), full.get("away")
    if home is None or away is None:
        return None
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def fetch_current_season_fdorg(season_label: str) -> list[dict]:
    url = f"{FDORG_BASE}/competitions/{FDORG_PL_CODE}/matches"
    resp = requests.get(url, headers=_fdorg_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for m in data.get("matches", []):
        utc_date = m.get("utcDate", "")[:10]  # YYYY-MM-DD
        score = m.get("score", {}).get("fullTime", {})
        home_goals = score.get("home")
        away_goals = score.get("away")
        rows.append({
            "season":     season_label,
            "date":       utc_date,
            "home_team":  m["homeTeam"]["shortName"],
            "away_team":  m["awayTeam"]["shortName"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result":     _fdorg_result(m),
            "matchday":   m.get("matchday"),
            "source":     "fdorg",
        })
    return rows


# ── OpenFootball (GitHub JSON, no key) ───────────────────────────────────────

OF_BASE = (
    "https://raw.githubusercontent.com/openfootball/football.json/master"
    "/{yy}-{zz}/en.1.json"
)


def fetch_current_season_openfootball(start_year: int) -> list[dict]:
    yy = str(start_year)[2:]
    zz = str(start_year + 1)[2:]
    url = OF_BASE.format(yy=yy, zz=zz)
    label = f"{start_year}/{zz}"

    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for match in data.get("matches", []):
        score = match.get("score", {})
        ft = score.get("ft")
        home_goals = ft[0] if ft else None
        away_goals = ft[1] if ft else None
        result = None
        if home_goals is not None and away_goals is not None:
            result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

        # parse matchday number from "Matchday N"
        matchday_str = match.get("round", "")
        matchday = None
        if matchday_str.startswith("Matchday"):
            try:
                matchday = int(matchday_str.split()[-1])
            except ValueError:
                pass

        rows.append({
            "season":     label,
            "date":       match.get("date"),
            "home_team":  match["team1"],
            "away_team":  match["team2"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result":     result,
            "matchday":   matchday,
            "source":     "openfootball",
        })
    return rows


# ── public entry point ───────────────────────────────────────────────────────

def sync_current_season(start_year: int | None = None) -> None:
    """
    Sync all fixtures (played + scheduled) for the current season into SQLite.
    Tries football-data.org first; falls back to OpenFootball if no API key is set.
    """
    import datetime as dt
    if start_year is None:
        today = dt.date.today()
        # EPL season starts in August; if we're before August, the current season
        # started last year
        start_year = today.year if today.month >= 8 else today.year - 1

    season_label = f"{start_year}/{str(start_year + 1)[-2:]}"
    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "")

    print(f"Syncing {season_label} …")
    if api_key:
        print("  Using football-data.org API")
        rows = fetch_current_season_fdorg(season_label)
    else:
        print("  No FOOTBALL_DATA_API_KEY set – using OpenFootball JSON")
        rows = fetch_current_season_openfootball(start_year)

    init_db()
    with get_connection() as conn:
        for row in rows:
            upsert_match(conn, row)
        conn.commit()

    played = sum(1 for r in rows if r["result"] is not None)
    scheduled = len(rows) - played
    print(f"  {played} played, {scheduled} scheduled – total {len(rows)} fixtures stored.")


if __name__ == "__main__":
    sync_current_season()
