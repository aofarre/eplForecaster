"""
Database setup and helper functions.

Schema
------
matches   – one row per played or scheduled fixture
standings – computed view: points tally per team per season
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "epl.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                season      TEXT    NOT NULL,   -- e.g. "2024/25"
                date        TEXT,               -- ISO-8601 YYYY-MM-DD
                home_team   TEXT    NOT NULL,
                away_team   TEXT    NOT NULL,
                home_goals  INTEGER,            -- NULL when not yet played
                away_goals  INTEGER,
                result      TEXT,               -- "H" | "D" | "A" | NULL
                matchday    INTEGER,
                source      TEXT,               -- "fdco" | "openfootball" | "fdorg"
                UNIQUE (season, date, home_team, away_team)
            );

            CREATE TABLE IF NOT EXISTS teams (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                short_name  TEXT                -- e.g. "Man City"
            );

            CREATE VIEW IF NOT EXISTS standings AS
            SELECT
                season,
                team,
                COUNT(*)                                            AS played,
                SUM(CASE WHEN result_for_team = 'W' THEN 1 ELSE 0 END) AS won,
                SUM(CASE WHEN result_for_team = 'D' THEN 1 ELSE 0 END) AS drawn,
                SUM(CASE WHEN result_for_team = 'L' THEN 1 ELSE 0 END) AS lost,
                SUM(goals_for)                                      AS gf,
                SUM(goals_against)                                  AS ga,
                SUM(goals_for) - SUM(goals_against)                 AS gd,
                SUM(CASE WHEN result_for_team = 'W' THEN 3
                         WHEN result_for_team = 'D' THEN 1
                         ELSE 0 END)                                AS points
            FROM (
                -- home perspective
                SELECT season, home_team AS team, home_goals AS goals_for,
                       away_goals AS goals_against,
                       CASE WHEN result = 'H' THEN 'W'
                            WHEN result = 'D' THEN 'D'
                            WHEN result = 'A' THEN 'L' END AS result_for_team
                FROM matches WHERE result IS NOT NULL
                UNION ALL
                -- away perspective
                SELECT season, away_team AS team, away_goals AS goals_for,
                       home_goals AS goals_against,
                       CASE WHEN result = 'A' THEN 'W'
                            WHEN result = 'D' THEN 'D'
                            WHEN result = 'H' THEN 'L' END AS result_for_team
                FROM matches WHERE result IS NOT NULL
            ) t
            GROUP BY season, team;
        """)
    print(f"Database ready at {DB_PATH}")


def upsert_match(conn: sqlite3.Connection, row: dict) -> None:
    """Insert or replace a match row."""
    conn.execute("""
        INSERT INTO matches
            (season, date, home_team, away_team, home_goals, away_goals, result, matchday, source)
        VALUES
            (:season, :date, :home_team, :away_team, :home_goals, :away_goals, :result, :matchday, :source)
        ON CONFLICT (season, date, home_team, away_team) DO UPDATE SET
            home_goals = excluded.home_goals,
            away_goals = excluded.away_goals,
            result     = excluded.result,
            matchday   = excluded.matchday,
            source     = excluded.source
    """, row)


if __name__ == "__main__":
    init_db()
