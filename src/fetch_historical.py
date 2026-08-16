"""
Fetch historical EPL results from football-data.co.uk and store in SQLite.

URL pattern: https://www.football-data.co.uk/mmz4281/{YYZZ}/E0.csv
  e.g. 2024/25 season → 2425/E0.csv

Key CSV columns used:
  Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR
"""

import csv
import io
import re
from datetime import datetime

import requests

from db import get_connection, init_db, upsert_match

BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

# EPL started 1992/93; go as far back as data allows
FIRST_SEASON_START = 1993  # start year of first available season (1993/94)


def season_code(start_year: int) -> str:
    """Convert start year to football-data.co.uk season code, e.g. 2024 → '2425'."""
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def season_label(start_year: int) -> str:
    """e.g. 2024 → '2024/25'."""
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def parse_date(raw: str) -> str | None:
    """Convert DD/MM/YYYY or DD/MM/YY to ISO-8601 YYYY-MM-DD."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def fetch_season(start_year: int) -> list[dict]:
    code = season_code(start_year)
    label = season_label(start_year)
    url = BASE_URL.format(code=code)

    resp = requests.get(url, timeout=15)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    rows = []
    for i, row in enumerate(reader):
        if not row.get("HomeTeam") or not row.get("FTHG"):
            continue  # skip incomplete rows (fixtures not yet played)
        rows.append({
            "season":      label,
            "date":        parse_date(row.get("Date", "")),
            "home_team":   row["HomeTeam"].strip(),
            "away_team":   row["AwayTeam"].strip(),
            "home_goals":  int(row["FTHG"]) if row["FTHG"].strip() else None,
            "away_goals":  int(row["FTAG"]) if row["FTAG"].strip() else None,
            "result":      row.get("FTR", "").strip() or None,
            "matchday":    None,  # not provided by this source
            "source":      "fdco",
        })
    return rows


def load_season(start_year: int, conn) -> int:
    rows = fetch_season(start_year)
    for row in rows:
        upsert_match(conn, row)
    return len(rows)


def load_all_historical(from_year: int = FIRST_SEASON_START, to_year: int | None = None) -> None:
    """
    Download and store all seasons from `from_year` up to (but not including)
    the current live season.  Defaults to every completed season since 1993/94.
    """
    import datetime as dt
    today = dt.date.today()
    if to_year is None:
        # "Current" live season started this August (or last August if before Aug).
        # All prior seasons are historical.
        current_season_start = today.year if today.month >= 8 else today.year - 1
        to_year = current_season_start - 1  # last *completed* season start year

    init_db()
    with get_connection() as conn:
        for year in range(from_year, to_year + 1):
            count = load_season(year, conn)
            label = season_label(year)
            if count:
                print(f"  {label}: {count} matches loaded")
            else:
                print(f"  {label}: no data (season may not exist)")
        conn.commit()
    print("Historical load complete.")


if __name__ == "__main__":
    load_all_historical()
