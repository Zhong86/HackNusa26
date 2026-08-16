"""
Penyimpanan riwayat sender asli pakai PostgreSQL — dicatat tiap kali email
masuk, dibaca lagi oleh lookup_sender_history(). Konfigurasi lewat env vars
(lihat bagian bawah file / .env.example).
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

PG_HOST = os.environ.get("POSTGRES_HOST", "localhost")
PG_PORT = os.environ.get("POSTGRES_PORT", "5432")
PG_DB = os.environ.get("POSTGRES_DB", "sentinel_loop")
PG_USER = os.environ.get("POSTGRES_USER", "postgres")
PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

_CONNINFO = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"


@contextmanager
def _connect():
    conn = psycopg.connect(_CONNINFO, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Buat tabel kalau belum ada. Dipanggil sekali saat app startup."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sender_events (
                id SERIAL PRIMARY KEY,
                sender TEXT NOT NULL,
                seen_at TIMESTAMPTZ NOT NULL,
                verdict TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sender ON sender_events(sender)")


def record_sender_event(sender: str, verdict: str | None = None) -> None:
    """Catat satu kemunculan sender. Dipanggil tiap email masuk lewat graph."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sender_events (sender, seen_at, verdict) VALUES (%s, %s, %s)",
            (sender, datetime.now(timezone.utc), verdict),
        )


def get_sender_history(sender: str) -> dict:
    """Baca riwayat asli sender ini dari DB — dipakai oleh lookup_sender_history()."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT seen_at, verdict FROM sender_events WHERE sender = %s ORDER BY seen_at ASC",
            (sender,),
        ).fetchall()

    if not rows:
        return {
            "sender": sender,
            "seen_before": False,
            "first_seen_days_ago": None,
            "prior_flag_count": 0,
        }

    first_seen = rows[0]["seen_at"]
    days_ago = (datetime.now(timezone.utc) - first_seen).days
    prior_flags = sum(1 for r in rows if r["verdict"] in ("quarantine", "escalate"))

    return {
        "sender": sender,
        "seen_before": True,
        "first_seen_days_ago": days_ago,
        "prior_flag_count": prior_flags,
    }
