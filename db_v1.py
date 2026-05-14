"""
QuorumAI — Database Layer (SQLite)
==================================
Auto-creates quorumai.db on first run.
All queries use parameterised statements to prevent SQL injection.
"""

import sqlite3
import os
import json
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quorumai.db")


@contextmanager
def _connection():
    """
    Context manager for SQLite connections.

    Replaces the old pattern:
        conn = _get_conn()
        conn.execute(...)
        conn.commit()
        conn.close()

    With:
        with _connection() as conn:
            conn.execute(...)

    Benefits:
      - Auto-commits on success, auto-rolls back on exception
      - Always closes the connection, even if the caller raises
      - Single source of truth for connection settings (row_factory, PRAGMA)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Backwards-compatible alias. Existing code that imports _get_conn keeps working.
def _get_conn():
    """Deprecated — prefer the _connection() context manager. Kept for back-compat."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _format_timestamp(ts: str) -> str:
    """Render a SQLite ISO timestamp as 'DD.MM.YYYY · HH:MM:SS', or fall back gracefully."""
    try:
        return datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S").strftime("%d.%m.%Y · %H:%M:%S")
    except Exception:
        return (ts or "")[:16]


def init_db():
    """Create tables if they don't exist. Safe to call on every app start."""
    with _connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                github_id INTEGER UNIQUE NOT NULL,
                email TEXT,
                display_name TEXT,
                avatar_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                provider_name TEXT NOT NULL,
                display_name TEXT,
                encrypted_key TEXT NOT NULL,
                model_id TEXT NOT NULL,
                endpoint_url TEXT NOT NULL,
                api_type TEXT DEFAULT 'openai_compat',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                summary TEXT NOT NULL,
                full_prompt TEXT NOT NULL,
                prompt TEXT,
                mode TEXT,
                quality INTEGER,
                complexity TEXT,
                synthesis TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_query_history_user ON query_history(user_id);

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                meta_prompt TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'New Project',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_projects_user ON projects(user_id);
        """)

    # ── Lightweight migrations: each column add is wrapped in its own
    # connection because a failed ALTER (column-already-exists) inside the
    # shared transaction above would abort the whole script.
    for column, ddl in [
        ("responses_json", "ALTER TABLE query_history ADD COLUMN responses_json TEXT"),
        ("project_id",     "ALTER TABLE query_history ADD COLUMN project_id TEXT"),
        ("extras_json",    "ALTER TABLE query_history ADD COLUMN extras_json TEXT"),
    ]:
        try:
            with _connection() as conn:
                conn.execute(ddl)
        except Exception:
            pass  # column already exists — that's fine


def save_history_entry(user_id, entry):
    """Persist a history entry to the database."""
    responses_json = json.dumps(entry.get("responses") or {})
    # Store all extra fields (errors, word_counts, analysis, model metadata) as one JSON blob
    extras = {
        "errors":           entry.get("errors") or {},
        "word_counts":      entry.get("word_counts") or {},
        "analysis":         entry.get("analysis"),
        "output_models":    entry.get("output_models") or [],
        "aggregator_model": entry.get("aggregator_model"),
        "analyzer_model":   entry.get("analyzer_model"),
        "models_used":      entry.get("models_used") or [],
        "successful":       entry.get("successful") or [],
        "failed":           entry.get("failed") or [],
        "times":            entry.get("times") or {},
    }
    extras_json = json.dumps(extras)
    with _connection() as conn:
        conn.execute(
            """INSERT INTO query_history
                   (user_id, summary, full_prompt, prompt, mode, quality, complexity, synthesis, responses_json, extras_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                entry.get("summary", ""),
                entry.get("full_prompt", ""),
                entry.get("prompt", ""),
                entry.get("mode", ""),
                entry.get("quality", 0),
                entry.get("complexity", ""),
                entry.get("synthesis"),
                responses_json,
                extras_json,
            ),
        )


def load_history(user_id, limit=50):
    """Load saved history entries for a user, oldest-first (matches session_state append order)."""
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM query_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    entries = []
    for row in rows:
        d = dict(row)
        ts_display = _format_timestamp(d.get("created_at", ""))
        try:
            responses = json.loads(d.get("responses_json") or "{}")
        except Exception:
            responses = {}
        try:
            extras = json.loads(d.get("extras_json") or "{}")
        except Exception:
            extras = {}
        entries.append({
            "timestamp":        ts_display,
            "summary":          d["summary"],
            "full_prompt":      d["full_prompt"],
            "prompt":           d["prompt"],
            "mode":             d["mode"],
            "quality":          d["quality"],
            "complexity":       d["complexity"],
            "synthesis":        d["synthesis"],
            "responses":        responses,
            # Prefer extras values; fall back gracefully for old rows
            "models_used":      extras.get("models_used") or list(responses.keys()),
            "successful":       extras.get("successful") or list(responses.keys()),
            "failed":           extras.get("failed") or [],
            "times":            extras.get("times") or {},
            "errors":           extras.get("errors") or {},
            "word_counts":      extras.get("word_counts") or {},
            "analysis":         extras.get("analysis"),
            "output_models":    extras.get("output_models") or [],
            "aggregator_model": extras.get("aggregator_model"),
            "analyzer_model":   extras.get("analyzer_model"),
        })
    return list(reversed(entries))  # oldest first


def clear_history(user_id):
    """Delete all history entries for a user."""
    with _connection() as conn:
        conn.execute("DELETE FROM query_history WHERE user_id = ?", (user_id,))


def save_meta_prompt(user_id, meta_prompt):
    """Upsert the user's meta-prompt."""
    with _connection() as conn:
        conn.execute(
            "INSERT INTO user_settings (user_id, meta_prompt) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET meta_prompt = excluded.meta_prompt",
            (user_id, meta_prompt or ""),
        )


def load_meta_prompt(user_id):
    """Return the user's saved meta-prompt, or empty string if none."""
    with _connection() as conn:
        row = conn.execute(
            "SELECT meta_prompt FROM user_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    return row["meta_prompt"] if row else ""


def get_or_create_user(github_id, email, display_name, avatar_url):
    """
    Find existing user by github_id, or create a new one.
    Returns a dict with user data.
    """
    with _connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE github_id = ?", (github_id,)).fetchone()
        if row:
            # Update last_login and any changed profile info
            conn.execute(
                "UPDATE users SET last_login = ?, email = ?, display_name = ?, avatar_url = ? WHERE github_id = ?",
                (datetime.now().isoformat(), email, display_name, avatar_url, github_id),
            )
            row = conn.execute("SELECT * FROM users WHERE github_id = ?", (github_id,)).fetchone()
        else:
            cursor = conn.execute(
                "INSERT INTO users (github_id, email, display_name, avatar_url, last_login) VALUES (?, ?, ?, ?, ?)",
                (github_id, email, display_name, avatar_url, datetime.now().isoformat()),
            )
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)


def get_user_by_id(user_id):
    """Fetch a user by their internal ID."""
    with _connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def save_project(user_id, project_id, title):
    """Upsert a project record."""
    with _connection() as conn:
        conn.execute(
            """INSERT INTO projects (id, user_id, title, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at""",
            (project_id, user_id, title[:65], datetime.now().isoformat()),
        )


def load_projects(user_id):
    """Return list of project dicts for a user, newest first."""
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def save_project_entry(user_id, project_id, entry):
    """Save a history entry tagged to a project."""
    responses_json = json.dumps(entry.get("responses") or {})
    with _connection() as conn:
        conn.execute(
            """INSERT INTO query_history
                   (user_id, project_id, summary, full_prompt, prompt, mode, quality, complexity, synthesis, responses_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, project_id,
                entry.get("summary", ""),
                entry.get("full_prompt", ""),
                entry.get("prompt", ""),
                entry.get("mode", ""),
                entry.get("quality", 0),
                entry.get("complexity", ""),
                entry.get("synthesis"),
                responses_json,
            ),
        )


def load_project_entries(user_id, project_id, limit=100):
    """Load history entries for one project, oldest first."""
    with _connection() as conn:
        rows = conn.execute(
            """SELECT * FROM query_history
               WHERE user_id = ? AND project_id = ?
               ORDER BY created_at ASC LIMIT ?""",
            (user_id, project_id, limit),
        ).fetchall()
    entries = []
    for row in rows:
        d = dict(row)
        ts_display = _format_timestamp(d.get("created_at", ""))
        try:
            responses = json.loads(d.get("responses_json") or "{}")
        except Exception:
            responses = {}
        entries.append({
            "timestamp": ts_display,
            "summary": d["summary"],
            "full_prompt": d["full_prompt"],
            "prompt": d["prompt"],
            "mode": d["mode"],
            "quality": d["quality"],
            "complexity": d["complexity"],
            "synthesis": d["synthesis"],
            "responses": responses,
            "models_used": list(responses.keys()),
        })
    return entries


def delete_project(user_id, project_id):
    """Delete a project and all its history entries."""
    with _connection() as conn:
        conn.execute("DELETE FROM query_history WHERE user_id = ? AND project_id = ?", (user_id, project_id))
        conn.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
