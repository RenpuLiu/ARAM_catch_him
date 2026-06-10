from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


RAW_DIRNAME = "raw"
MATCHES_CSV = "matches.csv"
PARTICIPANTS_CSV = "participants.csv"
MATCHES_SQLITE = "matches.sqlite"
SUMMARY_JSON = "summary_basic.json"


def ensure_data_dirs(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    (root / RAW_DIRNAME).mkdir(parents=True, exist_ok=True)
    return root


def save_json(path: str | Path, obj: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def write_matches_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    write_rows_csv(path, rows)


def write_rows_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_matches_csv(path: str | Path) -> list[dict[str, str]]:
    target = Path(path)
    if not target.exists() or target.stat().st_size == 0:
        return []

    with target.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_matches_sqlite(
    path: str | Path,
    rows: list[dict[str, Any]],
    participant_rows: list[dict[str, Any]] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(target) as conn:
        _write_table(conn, "matches", rows)
        _write_table(conn, "participants", participant_rows or [])
        conn.execute('CREATE INDEX IF NOT EXISTS idx_matches_id ON matches("match_id")')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_participants_match_id ON participants("match_id")'
        )
        if participant_rows and any("summoner_id" in row for row in participant_rows):
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_participants_summoner_id ON participants("summoner_id")'
            )


def raw_path(data_dir: str | Path, filename: str) -> Path:
    return Path(data_dir) / RAW_DIRNAME / filename


def matches_csv_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / MATCHES_CSV


def participants_csv_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / PARTICIPANTS_CSV


def matches_sqlite_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / MATCHES_SQLITE


def summary_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / SUMMARY_JSON


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _write_table(
    conn: sqlite3.Connection,
    table_name: str,
    rows: list[dict[str, Any]],
) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
    if not rows:
        conn.execute(f'CREATE TABLE "{table_name}" (match_id TEXT)')
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    columns = ", ".join(f'"{name}" TEXT' for name in fieldnames)
    placeholders = ", ".join("?" for _ in fieldnames)
    quoted = ", ".join(f'"{name}"' for name in fieldnames)

    conn.execute(f'CREATE TABLE "{table_name}" ({columns})')
    conn.executemany(
        f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})',
        [[_stringify(row.get(name)) for name in fieldnames] for row in rows],
    )
