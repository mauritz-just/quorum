"""
QuorumAI — Database Layer (SQLite)
==================================
Auto-creates quorumai.db on first run.
All queries use parameterised statements to prevent SQL injection.
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quorumai.db")


def _get_conn():
    """Get a database connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call on every app start."""
    conn = _get_conn()
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
    conn.commit()
    # Add responses_json column if it doesn't exist yet (migration for existing DBs)
    try:
        conn.execute("ALTER TABLE query_history ADD COLUMN responses_json TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists
    # Add project_id column if it doesn't exist yet (migration for existing DBs)
    try:
        conn.execute("ALTER TABLE query_history ADD COLUMN project_id TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists
    # Add extras_json column for errors, word_counts, analysis, model metadata
    try:
        conn.execute("ALTER TABLE query_history ADD COLUMN extras_json TEXT")
        conn.commit()
    except Exception:
        pass  # column already exists
    conn.close()


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
    conn = _get_conn()
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
    conn.commit()
    conn.close()


def load_history(user_id, limit=50):
    """Load saved history entries for a user, oldest-first (matches session_state append order)."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM query_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    entries = []
    for row in rows:
        d = dict(row)
        # Format timestamp: show date+time for past sessions, just time for today
        ts_full = d.get("created_at", "")
        try:
            from datetime import date
            ts_date = ts_full[:10]
            today = date.today().isoformat()
            ts_display = ts_full[11:19] if ts_date == today else ts_full[:16]
        except Exception:
            ts_display = ts_full[:16]
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
    conn = _get_conn()
    conn.execute("DELETE FROM query_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


def save_meta_prompt(user_id, meta_prompt):
    """Upsert the user's meta-prompt."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO user_settings (user_id, meta_prompt) VALUES (?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET meta_prompt = excluded.meta_prompt",
        (user_id, meta_prompt or ""),
    )
    conn.commit()
    conn.close()


def load_meta_prompt(user_id):
    """Return the user's saved meta-prompt, or empty string if none."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT meta_prompt FROM user_settings WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return row["meta_prompt"] if row else ""


def get_or_create_user(github_id, email, display_name, avatar_url):
    """
    Find existing user by github_id, or create a new one.
    Returns a dict with user data.
    """
    conn = _get_conn()
    cursor = conn.cursor()

    # Try to find existing user
    cursor.execute("SELECT * FROM users WHERE github_id = ?", (github_id,))
    row = cursor.fetchone()

    if row:
        # Update last_login and any changed profile info
        cursor.execute(
            "UPDATE users SET last_login = ?, email = ?, display_name = ?, avatar_url = ? WHERE github_id = ?",
            (datetime.now().isoformat(), email, display_name, avatar_url, github_id),
        )
        conn.commit()
        # Re-fetch to get updated data
        cursor.execute("SELECT * FROM users WHERE github_id = ?", (github_id,))
        row = cursor.fetchone()
    else:
        # Create new user
        cursor.execute(
            "INSERT INTO users (github_id, email, display_name, avatar_url, last_login) VALUES (?, ?, ?, ?, ?)",
            (github_id, email, display_name, avatar_url, datetime.now().isoformat()),
        )
        conn.commit()
        cursor.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,))
        row = cursor.fetchone()

    user = dict(row)
    conn.close()
    return user


def get_user_by_id(user_id):
    """Fetch a user by their internal ID."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def save_project(user_id, project_id, title):
    """Upsert a project record."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO projects (id, user_id, title, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET title=excluded.title, updated_at=excluded.updated_at""",
        (project_id, user_id, title[:65], datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def load_projects(user_id):
    """Return list of project dicts for a user, newest first."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_project_entry(user_id, project_id, entry):
    """Save a history entry tagged to a project."""
    responses_json = json.dumps(entry.get("responses") or {})
    conn = _get_conn()
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
    conn.commit()
    conn.close()


def load_project_entries(user_id, project_id, limit=100):
    """Load history entries for one project, oldest first."""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM query_history
           WHERE user_id = ? AND project_id = ?
           ORDER BY created_at ASC LIMIT ?""",
        (user_id, project_id, limit),
    )
    rows = cursor.fetchall()
    conn.close()
    entries = []
    for row in rows:
        d = dict(row)
        ts_full = d.get("created_at", "")
        try:
            from datetime import date
            ts_date = ts_full[:10]
            today = date.today().isoformat()
            ts_display = ts_full[11:19] if ts_date == today else ts_full[:16]
        except Exception:
            ts_display = ts_full[:16]
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
    conn = _get_conn()
    conn.execute("DELETE FROM query_history WHERE user_id = ? AND project_id = ?", (user_id, project_id))
    conn.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
    conn.commit()
    conn.close()
