"""
QuorumAI — API Key Manager
============================
Handles encryption, storage, retrieval, and validation of user API keys.
Keys are encrypted with Fernet (AES-128-CBC) before touching the database.
Decrypted keys only exist in RAM, never on disk or in logs.
"""

import hashlib
import os
import sqlite3
import requests
from datetime import datetime
from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

_MASTER_KEY = os.getenv("ENCRYPTION_MASTER_KEY", "")
try:
    _fernet = Fernet(_MASTER_KEY.encode()) if _MASTER_KEY else None
except Exception:
    _fernet = None

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quorumai.db")

# ──────────────────────────────────────────────
# Provider presets — auto-fill endpoint + model when user picks a provider
# ──────────────────────────────────────────────
PROVIDER_PRESETS = {
    "OpenAI": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o4-mini", "o5-mini"],
        "api_type": "openai_compat",
        "icon": "🟢",
    },
    "Anthropic": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "models": ["claude-sonnet-4-20250514", "claude-haiku-4-20250414"],
        "api_type": "anthropic",
        "icon": "🟠",
    },
    "Gemini": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
        "api_type": "gemini",
        "icon": "🔵",
    },
    "Groq": {
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "api_type": "openai_compat",
        "icon": "⚡",
    },
    "Mistral": {
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "models": ["mistral-large-latest", "devstral-medium-latest"],
        "api_type": "openai_compat",
        "icon": "▲",
    },
    "Cerebras": {
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "models": ["qwen-3-235b-a22b-instruct-2507", "llama-3.3-70b"],
        "api_type": "openai_compat",
        "icon": "◆",
    },
    "OpenRouter": {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4", "meta-llama/llama-3.3-70b-instruct:free"],
        "api_type": "openai_compat",
        "icon": "🔀",
    },
    "Custom": {
        "endpoint": "",
        "models": [],
        "api_type": "openai_compat",
        "icon": "🔧",
    },
}

MAX_ACTIVE_KEYS = 4   # max simultaneously active models
MAX_STORED_KEYS = None  # no storage limit
MAX_KEYS_PER_USER = MAX_ACTIVE_KEYS  # backward-compat alias

# ──────────────────────────────────────────────
# Model display name normalisation
# ──────────────────────────────────────────────
MODEL_DISPLAY_NAMES = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B",
    "llama-3.3-70b": "Llama 3.3 70B",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4.1": "GPT-4.1",
    "o4-mini": "o4 Mini",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-haiku-4-20250414": "Claude Haiku 4",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "mistral-large-latest": "Mistral Large",
    "devstral-medium-latest": "Devstral Medium",
    "qwen-3-235b-a22b-instruct-2507": "Qwen3 235B",
    "openai/gpt-4o": "GPT-4o",
    "anthropic/claude-sonnet-4": "Claude Sonnet 4",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3 70B (free)",
}


# Map legacy provider names → current display labels (for keys stored before a preset was renamed).
PROVIDER_DISPLAY_ALIASES = {
    "Google (Gemini)": "Gemini",
    "Google": "Gemini",
}


def get_model_display_name(model_id: str) -> str:
    """Return a human-readable model name, falling back to the raw model_id."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def normalize_provider_name(provider_name: str) -> str:
    """Map legacy / verbose provider names to the canonical display label."""
    return PROVIDER_DISPLAY_ALIASES.get(provider_name, provider_name)


def build_key_display_name(display_name: str, provider_name: str, model_id: str) -> str:
    """
    Canonical display name for a user key: 'Provider · ModelName · Alias'.
    All keys for the same model look identical except for the alias suffix.
    """
    provider_pretty = normalize_provider_name(provider_name)
    model_pretty = get_model_display_name(model_id)
    alias = (display_name or "").strip() or provider_pretty
    return f"{provider_pretty} · {model_pretty} · {alias}"


# ──────────────────────────────────────────────
# Encryption
# ──────────────────────────────────────────────
def encrypt_key(plaintext_key):
    """Encrypt an API key. Returns the ciphertext string."""
    if not _fernet:
        raise RuntimeError("ENCRYPTION_MASTER_KEY not set in .env")
    return _fernet.encrypt(plaintext_key.encode()).decode()


def decrypt_key(encrypted_key):
    """Decrypt an API key. Returns the plaintext string."""
    if not _fernet:
        raise RuntimeError("ENCRYPTION_MASTER_KEY not set in .env")
    return _fernet.decrypt(encrypted_key.encode()).decode()


def mask_key(plaintext_key):
    """Show only last 4 chars: sk-...7g2f"""
    if len(plaintext_key) <= 8:
        return "••••••••"
    return f"{plaintext_key[:3]}...{plaintext_key[-4:]}"


# ──────────────────────────────────────────────
# CRUD operations
# ──────────────────────────────────────────────
# Connection management is centralised in db.py — we import the same
# context manager here so there's exactly one source of truth for how the
# DB is opened (path, pragmas, row_factory).
from db import _connection


def save_key(user_id, provider_name, display_name, plaintext_key, model_id, endpoint_url, api_type="openai_compat"):
    """
    Encrypt and store an API key. Returns the new key's ID or raises ValueError on validation failure.

    Validations (all before touching the DB):
      - Total stored keys < MAX_STORED_KEYS
      - Display name is unique (case-insensitive, trimmed)
      - Plaintext key is unique (hash-based, no plaintext comparison)

    The new key is saved as active if the user is below MAX_ACTIVE_KEYS, otherwise inactive.
    """
    all_keys = get_keys(user_id)

    # Duplicate display name (case-insensitive, whitespace-normalised)
    norm_alias = (display_name or provider_name).strip().lower()
    for k in all_keys:
        existing_alias = (k["display_name"] or k["provider_name"]).strip().lower()
        if existing_alias == norm_alias:
            raise ValueError("An API with this name already exists.")

    # Duplicate key (hash-based — never compare plaintext directly)
    new_hash = hashlib.sha256(plaintext_key.strip().encode()).hexdigest()
    for k in all_keys:
        if k["key"] and hashlib.sha256(k["key"].strip().encode()).hexdigest() == new_hash:
            raise ValueError("This API key has already been added.")

    # Active status: on if below limit, off otherwise
    active_count = sum(1 for k in all_keys if k["is_active"])
    is_active = 1 if active_count < MAX_ACTIVE_KEYS else 0

    encrypted = encrypt_key(plaintext_key)
    with _connection() as conn:
        cursor = conn.execute(
            """INSERT INTO api_keys (user_id, provider_name, display_name, encrypted_key, model_id, endpoint_url, api_type, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, provider_name, display_name, encrypted, model_id, endpoint_url, api_type, is_active),
        )
        key_id = cursor.lastrowid
    return key_id, is_active


def get_keys(user_id):
    """
    Get all API keys for a user. Returns list of dicts.
    The 'key' field contains the DECRYPTED plaintext (handle with care).
    The 'masked_key' field is safe to display in UI.
    """
    with _connection() as conn:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()

    keys = []
    for row in rows:
        d = dict(row)
        try:
            plaintext = decrypt_key(d["encrypted_key"])
            d["key"] = plaintext
            d["masked_key"] = mask_key(plaintext)
        except Exception:
            d["key"] = None
            d["masked_key"] = "⚠️ decrypt error"
        # Remove the encrypted blob from the dict — no need to pass it around
        del d["encrypted_key"]
        keys.append(d)
    return keys


def delete_key(user_id, key_id):
    """Delete a key. Only works if the key belongs to the given user."""
    with _connection() as conn:
        conn.execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user_id))


def toggle_key(user_id, key_id):
    """Toggle a key's is_active status. No limit on how many can be active."""
    with _connection() as conn:
        row = conn.execute(
            "SELECT is_active FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user_id)
        ).fetchone()
        if not row:
            return
        conn.execute(
            "UPDATE api_keys SET is_active = CASE WHEN is_active = 1 THEN 0 ELSE 1 END WHERE id = ? AND user_id = ?",
            (key_id, user_id),
        )


def count_active_keys(user_id):
    """Count how many active keys a user has."""
    with _connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM api_keys WHERE user_id = ? AND is_active = 1", (user_id,)
        ).fetchone()[0]
    return count


def update_last_used(user_id, key_id):
    """Mark a key as recently used."""
    with _connection() as conn:
        conn.execute(
            "UPDATE api_keys SET last_used = ? WHERE id = ? AND user_id = ?",
            (datetime.now().isoformat(), key_id, user_id),
        )


# ──────────────────────────────────────────────
# Key validation — test if a key actually works
# ──────────────────────────────────────────────
def test_key(provider_name, plaintext_key, model_id, endpoint_url, api_type="openai_compat"):
    """
    Make a minimal API call to verify the key works.
    Returns (True, "OK") or (False, "error message").
    """
    try:
        if api_type == "gemini":
            return _test_gemini(plaintext_key, model_id, endpoint_url)
        else:
            return _test_openai_compat(plaintext_key, model_id, endpoint_url)
    except Exception as e:
        return False, str(e)


def _test_openai_compat(key, model_id, endpoint_url):
    """Test an OpenAI-compatible API key with a tiny request."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
    }
    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=15)
    if resp.status_code == 200:
        return True, "Key is valid"
    else:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


def _test_gemini(key, model_id, endpoint_url):
    """Test a Gemini API key with a tiny request."""
    url = endpoint_url.format(model=model_id)
    url = f"{url}?key={key}"
    payload = {
        "contents": [{"parts": [{"text": "Hi"}]}],
        "generationConfig": {"maxOutputTokens": 5},
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code == 200:
        return True, "Key is valid"
    else:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"


# ──────────────────────────────────────────────
# Build MODELS dict from user keys + fallbacks
# ──────────────────────────────────────────────
def build_models_dict(user_keys, fallback_models):
    """
    Merge user's own API keys with the app's built-in fallback models.
    User keys appear first, marked with 🔑.
    Returns a dict compatible with the existing MODELS format.
    """
    models = {}

    # Add user's own keys (active only)
    for k in user_keys:
        if not k["is_active"] or not k["key"]:
            continue
        name = build_key_display_name(k["display_name"], k["provider_name"], k["model_id"])
        models[name] = {
            "api_key": k["key"],
            "endpoint": k["endpoint_url"],
            "model_id": k["model_id"],
            "type": k["api_type"],
            "icon": "🔑",
            "color": "#10B981",
            "provider": k["provider_name"],
            "user_key_id": k["id"],  # track which DB key this is
        }

    # Add fallback models
    for name, cfg in fallback_models.items():
        models[name] = cfg

    return models
