"""
Fetch EPL squad market values from the transfermarkt-datasets CDN.

Source: https://github.com/dcaribou/transfermarkt-datasets
CDN:    https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/

The `clubs.csv.gz` table includes:
  club_id, name, squad_size, average_age, foreigners_number,
  domestic_competition_id, total_market_value, ...

We filter to domestic_competition_id == 'GB1' (Premier League).

The data is refreshed weekly so it reflects current summer transfer window
activity — a useful predictor of relative team strength at season start.

Squad values are stored in the `squad_values` table:
  (club_name TEXT, season TEXT, total_value_eur REAL, source TEXT)
"""

import gzip
import io
import sqlite3
from pathlib import Path

import requests

from db import DB_PATH, get_connection, init_db

CDN_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
CLUBS_URL = f"{CDN_BASE}/clubs.csv.gz"

EPL_COMPETITION_ID = "GB1"


# ── DB migration: add squad_values table if not present ──────────────────────

def ensure_squad_values_table() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS squad_values (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                club_name   TEXT    NOT NULL,
                season      TEXT    NOT NULL,       -- e.g. "2026/27"
                total_value_eur REAL,
                squad_size  INTEGER,
                source      TEXT,
                fetched_at  TEXT DEFAULT (date('now')),
                UNIQUE (club_name, season)
            )
        """)


# ── Name normalisation ────────────────────────────────────────────────────────

# Transfermarkt names → canonical names used in our matches table.
# Add entries as needed.
TM_NAME_MAP: dict[str, str] = {
    "Arsenal FC":                  "Arsenal",
    "Manchester City FC":          "Man City",
    "Manchester United FC":        "Man United",
    "Aston Villa FC":              "Aston Villa",
    "Liverpool FC":                "Liverpool",
    "AFC Bournemouth":             "Bournemouth",
    "Sunderland AFC":              "Sunderland",
    "Brighton & Hove Albion FC":   "Brighton",
    "Brentford FC":                "Brentford",
    "Chelsea FC":                  "Chelsea",
    "Fulham FC":                   "Fulham",
    "Newcastle United FC":         "Newcastle",
    "Everton FC":                  "Everton",
    "Leeds United FC":             "Leeds",
    "Crystal Palace FC":           "Crystal Palace",
    "Nottingham Forest FC":        "Nott'm Forest",
    "Tottenham Hotspur FC":        "Tottenham",
    "Coventry City FC":            "Coventry",
    "Ipswich Town FC":             "Ipswich",
    "Hull City AFC":               "Hull",
    # Relegated (may appear in historical data)
    "West Ham United FC":          "West Ham",
    "Wolverhampton Wanderers FC":  "Wolves",
    "Burnley FC":                  "Burnley",
}


def normalise_name(tm_name: str) -> str:
    return TM_NAME_MAP.get(tm_name, tm_name)


# ── Fetch & parse ─────────────────────────────────────────────────────────────

def fetch_current_squad_values() -> list[dict]:
    """
    Download clubs.csv.gz from the transfermarkt-datasets CDN and return
    EPL clubs with their current total market value (EUR).
    """
    resp = requests.get(CLUBS_URL, timeout=30)
    resp.raise_for_status()

    # Decompress and parse CSV
    import csv
    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("domestic_competition_id") == EPL_COMPETITION_ID]

    result = []
    for r in rows:
        raw_value = r.get("total_market_value", "").strip()
        value_eur: float | None = None
        if raw_value and raw_value.lower() not in ("", "none", "null"):
            try:
                value_eur = float(raw_value)
            except ValueError:
                pass

        result.append({
            "club_name":    normalise_name(r.get("name", "").strip()),
            "total_value_eur": value_eur,
            "squad_size":   int(r["squad_size"]) if r.get("squad_size", "").strip().isdigit() else None,
        })

    return result


def sync_squad_values(season: str) -> None:
    """
    Fetch current EPL squad values and store them against `season`.
    Call this once per summer transfer window (e.g. before the season starts).
    """
    init_db()
    ensure_squad_values_table()

    print(f"Fetching squad values for {season} …")
    clubs = fetch_current_squad_values()

    with get_connection() as conn:
        for club in clubs:
            conn.execute("""
                INSERT INTO squad_values (club_name, season, total_value_eur, squad_size, source)
                VALUES (?, ?, ?, ?, 'transfermarkt-datasets')
                ON CONFLICT (club_name, season) DO UPDATE SET
                    total_value_eur = excluded.total_value_eur,
                    squad_size      = excluded.squad_size,
                    fetched_at      = date('now')
            """, (club["club_name"], season, club["total_value_eur"], club["squad_size"]))
        conn.commit()

    print(f"  Stored {len(clubs)} clubs for {season}.")
    for c in sorted(clubs, key=lambda x: -(x["total_value_eur"] or 0)):
        val = f"€{c['total_value_eur']/1e6:.0f}m" if c["total_value_eur"] else "n/a"
        print(f"  {c['club_name']:<30} {val}")


def get_squad_values(season: str) -> dict[str, float]:
    """
    Return a {club_name: total_value_eur} dict for the given season.
    Returns an empty dict if no data has been synced yet.
    """
    ensure_squad_values_table()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT club_name, total_value_eur FROM squad_values WHERE season = ?",
            (season,)
        ).fetchall()
    return {r["club_name"]: r["total_value_eur"] for r in rows if r["total_value_eur"]}


if __name__ == "__main__":
    import sys
    season_arg = sys.argv[1] if len(sys.argv) > 1 else "2026/27"
    sync_squad_values(season_arg)
