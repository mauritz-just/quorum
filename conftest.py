"""
Shared pytest fixtures for the QuorumAI test suite.

The `temp_db` fixture creates an isolated, encrypted SQLite database for each
test that needs DB access, monkey-patches DB_PATH inside both db.py and
key_manager.py so the production modules don't write to the real quorumai.db,
and sets a one-shot ENCRYPTION_MASTER_KEY in the environment.

Tests that don't touch the DB (PromptAnalyzer, ModelClient with mocked HTTP)
don't need this fixture.
"""
import base64
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make the project root importable so `import db`, `import key_manager`, etc. work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_db(monkeypatch):
    """
    Create a fresh, isolated SQLite database for one test.

    Yields the path to the temp DB. The same path is patched into both
    db.DB_PATH and key_manager.DB_PATH so production code uses the temp file.
    The file is deleted after the test runs.
    """
    # Each test gets its own Fernet key so encryption is exercised end-to-end
    # without leaking keys between tests.
    monkeypatch.setenv("ENCRYPTION_MASTER_KEY", base64.urlsafe_b64encode(os.urandom(32)).decode())

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()

    # Import lazily so the env var is set first.
    import db
    import key_manager

    monkeypatch.setattr(db, "DB_PATH", tmp.name)
    monkeypatch.setattr(key_manager, "DB_PATH", tmp.name)
    # key_manager creates its Fernet at import time from the (then-empty) env var,
    # so rebuild it now that ENCRYPTION_MASTER_KEY is set.
    from cryptography.fernet import Fernet
    monkeypatch.setattr(key_manager, "_fernet", Fernet(os.environ["ENCRYPTION_MASTER_KEY"].encode()))

    db.init_db()
    yield tmp.name

    try:
        os.unlink(tmp.name)
    except OSError:
        pass


@pytest.fixture
def test_user(temp_db):
    """A freshly-created user for tests that need a valid user_id."""
    import db
    return db.get_or_create_user(
        github_id=99999,
        email="pytest@example.com",
        display_name="Pytest User",
        avatar_url="",
    )


@pytest.fixture(scope="session")
def app_module():
    """
    Load PromptAnalyzer, ModelClient, and friends from appv15.py *without*
    executing the Streamlit UI code at the bottom of the file.

    appv15.py calls require_auth() at module load time, which would block
    the test process indefinitely. Instead we read the source and exec only
    the safe portion (up to but not including the first call to require_auth).
    The result is returned as a namespace dict so tests can do
    `app_module["PromptAnalyzer"]`, etc.
    """
    import types
    from unittest.mock import MagicMock

    app_path = PROJECT_ROOT / "appv15.py"
    src_lines = app_path.read_text(encoding="utf-8").split("\n")

    # Find the line where the UI section begins (call to require_auth at module level).
    # We exec everything before that line. This keeps the analyzer + ModelClient
    # definitions available but stops short of any Streamlit page rendering.
    cutoff = next(
        (i for i, line in enumerate(src_lines) if line.startswith("user = require_auth()")),
        len(src_lines),
    )
    safe_src = "\n".join(src_lines[:cutoff])

    # Stub streamlit so any module-level st.* calls in the safe portion are no-ops.
    sys.modules.setdefault("streamlit", MagicMock())

    ns = {"__name__": "app_under_test"}
    exec(compile(safe_src, str(app_path), "exec"), ns)
    return ns
