"""
QuorumAI — Database Layer (SQLite)
==================================
Auto-creates quorumai.db on first run.
All queries use parameterised statements to prevent SQL injection.
"""

import sqlite3
import os
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
    """)
    conn.commit()
    conn.close()


def save_history_entry(user_id, entry):
    """Persist a history entry to the database."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO query_history (user_id, summary, full_prompt, prompt, mode, quality, complexity, synthesis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            entry.get("summary", ""),
            entry.get("full_prompt", ""),
            entry.get("prompt", ""),
            entry.get("mode", ""),
            entry.get("quality", 0),
            entry.get("complexity", ""),
            entry.get("synthesis"),
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
        entries.append({
            "timestamp": ts_display,
            "summary": d["summary"],
            "full_prompt": d["full_prompt"],
            "prompt": d["prompt"],
            "mode": d["mode"],
            "quality": d["quality"],
            "complexity": d["complexity"],
            "synthesis": d["synthesis"],
            "models_used": [],
            "successful": [],
            "failed": [],
            "times": {},
            "responses": {},
        })
    return list(reversed(entries))  # oldest first


def clear_history(user_id):
    """Delete all history entries for a user."""
    conn = _get_conn()
    conn.execute("DELETE FROM query_history WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


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
