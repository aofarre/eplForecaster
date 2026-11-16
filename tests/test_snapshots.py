import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from src.snapshots import save_snapshot, scheduled_snapshot_date

UTC = dt.timezone.utc


class ScheduleTests(unittest.TestCase):
    def test_timezone_and_daylight_saving(self):
        cases = [
            ("2026-03-02T23:00:00+00:00", "0 23 * * 1", "2026-03-02"),
            ("2026-03-09T22:00:00+00:00", "0 22 * * 1", "2026-03-09"),
            ("2026-10-26T22:00:00+00:00", "0 22 * * 1", "2026-10-26"),
            ("2026-11-02T23:00:00+00:00", "0 23 * * 1", "2026-11-02"),
        ]
        for now, cron, expected in cases:
            with self.subTest(now=now):
                result = scheduled_snapshot_date(cron, dt.datetime.fromisoformat(now))
                self.assertEqual(result.isoformat(), expected)

    def test_other_trigger_early_and_late_runs_are_skipped(self):
        cases = [
            ("2026-01-05T22:00:00+00:00", "0 22 * * 1"),
            ("2026-07-06T23:00:00+00:00", "0 23 * * 1"),
            ("2026-07-06T21:59:00+00:00", "0 22 * * 1"),
            ("2026-07-07T04:00:00+00:00", "0 22 * * 1"),
            ("2026-07-06T22:30:00+00:00", "30 22 * 8-12,1-5 *"),
        ]
        for now, cron in cases:
            with self.subTest(now=now, cron=cron):
                self.assertIsNone(scheduled_snapshot_date(cron, dt.datetime.fromisoformat(now)))
        self.assertEqual(
            scheduled_snapshot_date("0 22 * * 1", dt.datetime(2026, 7, 7, 3, 59, tzinfo=UTC)),
            dt.date(2026, 7, 6),
        )


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.docs = Path(self.temp.name)
        (self.docs / "snapshots").mkdir()
        (self.docs / "snapshots" / "index.json").write_text("[]\n")
        self.forecast = {"season": "2026/27", "updated_at": "2026-11-02T23:05:00Z", "teams": [{"team": "Arsenal"}]}
        self.write_forecast()

    def write_forecast(self):
        (self.docs / "data.json").write_text(json.dumps(self.forecast))

    def test_archive_preserves_forecast_and_rerun_does_not_overwrite(self):
        date = dt.date(2026, 11, 2)
        captured = dt.datetime(2026, 11, 2, 23, 6, tzinfo=UTC)
        self.assertTrue(save_snapshot(self.docs, date, captured))
        archive = self.docs / "snapshots"
        original = (archive / "2026-11-02.json").read_bytes()
        entry = json.loads((archive / "index.json").read_text())[0]
        self.assertEqual(json.loads(original), self.forecast)
        self.assertEqual(entry["scheduled_at"], "2026-11-02T18:00:00-05:00")
        self.assertEqual(entry["captured_at"], "2026-11-02T23:06:00Z")
        self.forecast["teams"] = [{"team": "Chelsea"}]
        self.write_forecast()
        self.assertFalse(save_snapshot(self.docs, date, captured))
        self.assertEqual((archive / "2026-11-02.json").read_bytes(), original)
        self.assertEqual(len(json.loads((archive / "index.json").read_text())), 1)

    def test_multiple_weeks_sorted_and_inconsistent_archive_fails(self):
        captured = dt.datetime(2026, 11, 9, 23, tzinfo=UTC)
        save_snapshot(self.docs, dt.date(2026, 11, 2), captured)
        save_snapshot(self.docs, dt.date(2026, 11, 9), captured)
        index_path = self.docs / "snapshots" / "index.json"
        entries = json.loads(index_path.read_text())
        self.assertEqual([x["path"] for x in entries], ["snapshots/2026-11-09.json", "snapshots/2026-11-02.json"])
        index_path.write_text("[]")
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            save_snapshot(self.docs, dt.date(2026, 11, 2), captured)

    def test_invalid_forecast_and_date_fail(self):
        self.forecast["teams"] = []
        self.write_forecast()
        with self.assertRaisesRegex(ValueError, "without teams"):
            save_snapshot(self.docs, dt.date(2026, 11, 2), dt.datetime.now(UTC))
        with self.assertRaisesRegex(ValueError, "Monday"):
            save_snapshot(self.docs, dt.date(2026, 11, 3), dt.datetime.now(UTC))


if __name__ == "__main__":
    unittest.main()
