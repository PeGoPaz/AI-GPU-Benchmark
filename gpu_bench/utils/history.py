"""Every exported run, indexed in one SQLite database.

The reports are scattered: logs/ by default, a sweep directory per sweep,
wherever --output-dir pointed. The history keeps one row per report — the
figures runs are compared by, plus the whole report as JSON — in a
database in the user's data directory, so "every run on this card" is a
query instead of a search through folders. The files stay the record; the
history only points at them, and rebuilding it is an import away.

The schema is versioned with SQLite's own user_version pragma: each entry
in MIGRATIONS moves the database up one version, in a transaction, so a
database written by an older version of the app is brought up to date the
first time a newer one opens it — and never half-migrated.
"""
import glob
import json
import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

APP_DIR = "AI-GPU-Benchmark"

# Index i takes the database from version i to version i + 1.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE runs (
        id INTEGER PRIMARY KEY,
        report_path TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        app_version TEXT,
        gpu TEXT,
        model TEXT,
        precision TEXT,
        batch_size INTEGER,
        seq_len INTEGER,
        gradient_checkpointing INTEGER,
        total_steps INTEGER,
        aborted INTEGER,
        tokens_per_sec REAL,
        tokens_per_joule REAL,
        est_tflops REAL,
        mfu REAL,
        avg_power_w REAL,
        max_temp_c REAL,
        report TEXT NOT NULL
    );
    CREATE INDEX runs_by_time ON runs (created_at);
    CREATE INDEX runs_by_card ON runs (gpu, model);
    """,
]

# Columns copied out of a report, by where they live in it.
_COLUMNS: dict[str, tuple[str, str]] = {
    "app_version": ("", "app_version"),
    "gpu": ("gpu", "name"),
    "model": ("workload", "model"),
    "precision": ("workload", "precision"),
    "batch_size": ("workload", "batch_size"),
    "seq_len": ("workload", "seq_len"),
    "gradient_checkpointing": ("workload", "gradient_checkpointing"),
    "total_steps": ("results", "total_steps"),
    "aborted": ("results", "aborted"),
    "tokens_per_sec": ("results", "tokens_per_sec"),
    "tokens_per_joule": ("results", "tokens_per_joule"),
    "est_tflops": ("results", "est_tflops"),
    "mfu": ("results", "mfu"),
    "avg_power_w": ("results", "avg_power_w"),
    "max_temp_c": ("results", "max_temp_c"),
}


def data_dir() -> str:
    """Where the history lives: GPU_BENCH_HOME, or the platform's user data dir."""
    override = os.environ.get("GPU_BENCH_HOME")
    if override:
        return override
    # Through a plain str: mypy evaluates sys.platform checks for the platform
    # it runs on, and would call the other two branches unreachable.
    system: str = sys.platform
    if system == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, APP_DIR)
    if system == "darwin":
        return os.path.expanduser(f"~/Library/Application Support/{APP_DIR}")
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, APP_DIR.lower())


def default_path() -> str:
    return os.path.join(data_dir(), "history.sqlite3")


@dataclass(frozen=True)
class HistoryRow:
    report_path: str
    created_at: str
    gpu: str | None
    model: str | None
    precision: str | None
    batch_size: int | None
    seq_len: int | None
    tokens_per_sec: float | None
    tokens_per_joule: float | None
    mfu: float | None
    max_temp_c: float | None

    @property
    def html_path(self) -> str | None:
        """The run's HTML page, if it was written beside the report."""
        candidate = self.report_path[:-len(".json")] + ".html"
        return candidate if os.path.isfile(candidate) else None


_ROW_FIELDS = ("report_path", "created_at", "gpu", "model", "precision", "batch_size",
               "seq_len", "tokens_per_sec", "tokens_per_joule", "mfu", "max_temp_c")


class History:
    """The database, opened and brought up to the current schema."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_path()
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        # Autocommit at the driver level: transactions are begun and ended
        # explicitly below, and the module's implicit ones would collide.
        self.db = sqlite3.connect(self.path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        try:
            self.migrate()
        except BaseException:
            # Closed here, or Windows would keep the file locked until the
            # connection was garbage-collected.
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def version(self) -> int:
        return int(self.db.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> None:
        """Applies every migration past the database's version, each atomically."""
        if self.version > len(MIGRATIONS):
            raise RuntimeError(f"{self.path} has schema version {self.version}, newer "
                               f"than this app knows ({len(MIGRATIONS)}); upgrade the app")
        for version in range(self.version, len(MIGRATIONS)):
            with self._transaction():
                for statement in _statements(MIGRATIONS[version]):
                    self.db.execute(statement)
                # PRAGMA cannot be parameterised; the value is our own int.
                self.db.execute(f"PRAGMA user_version = {version + 1}")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def record(self, report_path: str, report: dict[str, Any]) -> None:
        """Adds a run, or refreshes it if this report is already in the history."""
        values = {"report_path": os.path.abspath(report_path),
                  "created_at": report.get("created_at") or "",
                  "report": json.dumps(report, ensure_ascii=False)}
        for column, (section, key) in _COLUMNS.items():
            source = report if not section else (report.get(section) or {})
            value = source.get(key)
            values[column] = int(value) if isinstance(value, bool) else value
        columns = ", ".join(values)
        placeholders = ", ".join(f":{c}" for c in values)
        updates = ", ".join(f"{c} = excluded.{c}" for c in values if c != "report_path")
        with self._transaction():
            self.db.execute(f"INSERT INTO runs ({columns}) VALUES ({placeholders}) "
                            f"ON CONFLICT (report_path) DO UPDATE SET {updates}", values)

    def runs(self, limit: int | None = None) -> list[HistoryRow]:
        """Newest first."""
        query = f"SELECT {', '.join(_ROW_FIELDS)} FROM runs ORDER BY created_at DESC, id DESC"
        params: tuple[int, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        return [HistoryRow(*row) for row in self.db.execute(query, params)]

    def report(self, report_path: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT report FROM runs WHERE report_path = ?",
                              (os.path.abspath(report_path),)).fetchone()
        return json.loads(row[0]) if row else None

    def remove(self, report_path: str) -> None:
        """Forgets a run. Its files are left exactly where they are."""
        with self._transaction():
            self.db.execute("DELETE FROM runs WHERE report_path = ?",
                            (os.path.abspath(report_path),))

    def import_dir(self, directory: str) -> tuple[int, list[str]]:
        """Records every run report under a directory; returns (added, unreadable)."""
        added, unreadable = 0, []
        pattern = os.path.join(directory, "**", "report_*.json")
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(path, encoding="utf-8") as f:
                    report = json.load(f)
                if not isinstance(report, dict) or "results" not in report:
                    raise ValueError("not a run report")
            except (OSError, ValueError) as e:
                unreadable.append(f"{path}: {e}")
                continue
            self.record(path, report)
            added += 1
        return added, unreadable


def _statements(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]


def record_run(report_path: str, report: dict[str, Any]) -> str | None:
    """Records one exported run; returns why not, rather than raising.

    The history is a convenience on top of the files: a locked or unwritable
    database must never cost a run its export.
    """
    try:
        with History() as history:
            history.record(report_path, report)
    except (sqlite3.Error, OSError, RuntimeError) as e:
        return f"run not added to the history: {e}"
    return None
