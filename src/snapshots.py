"""Schedule and persist weekly static forecast snapshots."""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
SNAPSHOT_CRONS = {"0 22 * * 1", "0 23 * * 1"}


def scheduled_snapshot_date(schedule: str, now: dt.datetime) -> dt.date | None:
    """Return the eligible Monday for a snapshot trigger, or None."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if schedule not in SNAPSHOT_CRONS:
        return None
    local = now.astimezone(EASTERN)
    if local.weekday() != 0:
        return None
    target = dt.datetime.combine(local.date(), dt.time(18), EASTERN)
    expected_cron = f"0 {target.astimezone(dt.timezone.utc).hour} * * 1"
    if schedule != expected_cron or local < target:
        return None
    return local.date()


def save_snapshot(docs: Path, date: dt.date, captured_at: dt.datetime) -> bool:
    """Save a forecast once per Monday. Return False if already archived."""
    if date.weekday() != 0:
        raise ValueError("snapshot date must be a Monday")
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")

    archive = docs / "snapshots"
    index_path = archive / "index.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))
    filename = f"{date.isoformat()}.json"
    snapshot_path = archive / filename
    relative_path = f"snapshots/{filename}"
    existing = [entry for entry in entries if entry["path"] == relative_path]
    if len(existing) > 1 or bool(existing) != snapshot_path.exists():
        raise ValueError(f"Inconsistent snapshot archive for {date}")
    if existing:
        print(f"Snapshot for {date} already exists; leaving it unchanged.")
        return False

    forecast_path = docs / "data.json"
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    # Require the fields used by the selector and the forecast table.
    if not isinstance(forecast["teams"], list) or not forecast["teams"]:
        raise ValueError("Cannot archive a forecast without teams")
    entry = {
        "path": relative_path,
        "scheduled_at": dt.datetime.combine(date, dt.time(18), EASTERN).isoformat(),
        "captured_at": captured_at.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "season": forecast["season"],
        "updated_at": forecast["updated_at"],
    }
    entries.append(entry)
    entries.sort(key=lambda item: item["scheduled_at"], reverse=True)
    shutil.copyfile(forecast_path, snapshot_path)
    index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"Archived forecast for {date} in {snapshot_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--schedule", required=True)
    save = sub.add_parser("save")
    save.add_argument("--date", type=dt.date.fromisoformat, required=True)
    save.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    now = dt.datetime.now(dt.timezone.utc)
    if args.command == "check":
        date = scheduled_snapshot_date(args.schedule, now)
        print(date.isoformat() if date else "")
    else:
        save_snapshot(args.docs, args.date, now)


if __name__ == "__main__":
    main()
