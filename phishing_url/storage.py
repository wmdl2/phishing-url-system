"""使用本地 SQLite 持久化；不保存网址查询参数的原始值。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def initialize(path: Path) -> None:
    with _connect(path) as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                version TEXT PRIMARY KEY,
                trained_at TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                test_metrics_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL REFERENCES model_versions(version),
                checked_at TEXT NOT NULL DEFAULT (datetime('now')),
                display_url TEXT NOT NULL,
                status TEXT NOT NULL,
                risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
                source TEXT NOT NULL CHECK (source IN ('single', 'batch'))
            );
            CREATE INDEX IF NOT EXISTS idx_predictions_checked_at
                ON predictions(checked_at DESC);
            """
        )


def register_model(path: Path, metadata: dict) -> None:
    initialize(path)
    with _connect(path) as db:
        db.execute(
            """INSERT OR IGNORE INTO model_versions
               (version, trained_at, parameters_json, test_metrics_json)
               VALUES (?, ?, ?, ?)""",
            (
                metadata["model_version"],
                metadata["trained_at"],
                json.dumps(metadata["selected_parameters"], ensure_ascii=False),
                json.dumps(metadata["test_metrics"], ensure_ascii=False),
            ),
        )


def save_prediction(path: Path, prediction: dict, source: str) -> None:
    if source not in {"single", "batch"}:
        raise ValueError("来源必须是 single 或 batch")
    with _connect(path) as db:
        db.execute(
            """INSERT INTO predictions
               (model_version, display_url, status, risk_score, source)
               VALUES (?, ?, ?, ?, ?)""",
            (
                prediction["model_version"],
                prediction["display_url"],
                prediction["status"],
                prediction["risk_score"],
                source,
            ),
        )


def recent_predictions(path: Path, limit: int = 100, status: str | None = None) -> list[dict]:
    query = (
        "SELECT id, checked_at, display_url, status, risk_score, source, "
        "model_version FROM predictions"
    )
    values: list[object] = []
    if status:
        query += " WHERE status = ?"
        values.append(status)
    query += " ORDER BY id DESC LIMIT ?"
    values.append(limit)
    with _connect(path) as db:
        return [dict(row) for row in db.execute(query, values).fetchall()]


def count_predictions(path: Path) -> int:
    with _connect(path) as db:
        return int(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
