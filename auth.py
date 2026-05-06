"""
QuorumAI — Authentication Layer (GitHub OAuth)
===============================================
Handles the full OAuth flow within Streamlit:
  1. Generate GitHub login URL
  2. Handle the callback (code → token → user info)
  3. Session management via st.session_state

No external auth services needed — talks directly to GitHub's API.
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv
from db import get_or_create_user

load_dotenv()

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# GitHub OAuth endpoints
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


def get_login_url():
    """Build the GitHub OAuth authorization URL."""
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "scope": "read:user user:email",
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{GITHUB_AUTH_URL}?{query}"


def _exchange_code_for_token(code):
    """Exchange the OAuth code for an access token (server-to-server)."""
    resp = requests.post(
        GITHUB_TOKEN_URL,
        headers={"Accept": "application/json"},
        data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"GitHub OAuth error: {data.get('error_description', data['error'])}")

    return data["access_token"]


def _fetch_github_user(token):
    """Fetch the authenticated user's profile + primary email from GitHub API."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    # Get profile
    resp = requests.get(GITHUB_USER_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    profile = resp.json()

    # Get primary email (may not be public on profile)
    email = profile.get("email")
    if not email:
        resp = requests.get(GITHUB_EMAILS_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        emails = resp.json()
        primary = next((e for e in emails if e.get("primary")), None)
        email = primary["email"] if primary else emails[0]["email"] if emails else None

    return {
        "github_id": profile["id"],
        "email": email,
        "display_name": profile.get("name") or profile.get("login", "User"),
        "avatar_url": profile.get("avatar_url", ""),
        "github_username": profile.get("login", ""),
    }


def handle_callback(code):
    """
    Complete the OAuth flow: code → token → GitHub profile → local DB user.
    Returns the user dict, or None on failure.
    """
    try:
        token = _exchange_code_for_token(code)
        github_user = _fetch_github_user(token)
        user = get_or_create_user(
            github_id=github_user["github_id"],
            email=github_user["email"],
            display_name=github_user["display_name"],
            avatar_url=github_user["avatar_url"],
        )
        # Store github username separately (not in DB, just for display)
        user["github_username"] = github_user["github_username"]
        return user
    except Exception as e:
        st.error(f"Login failed: {e}")
        return None


def check_session():
    """Check if the user is currently logged in. Returns True/False."""
    return st.session_state.get("user") is not None


def logout():
    """Clear the user session."""
    st.session_state.pop("user", None)
    st.session_state.pop("user_keys", None)


def require_auth():
    """
    Call this at the top of app.py. Shows login screen if not authenticated.
    Returns the user dict if logged in.

    Usage:
        user = require_auth()
        if not user:
            st.stop()
    """
    # Handle OAuth callback if GitHub just redirected back
    params = st.query_params
    code = params.get("code")
    if code:
        user = handle_callback(code)
        if user:
            st.session_state.user = user
            st.query_params.clear()
            st.rerun()

    # Check if already logged in
    if check_session():
        return st.session_state.user

    # Show login screen
    _show_login_page()
    return None


def _show_login_page():
    """Render the login screen."""
    st.markdown("""
    <style>
        .login-box {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            padding: 2.5rem;
            border-radius: 16px;
            text-align: center;
            margin: 2rem auto;
            max-width: 420px;
        }
        .login-box h1 { color: #fff; font-size: 1.8rem; margin-bottom: 0.5rem; }
        .login-box p { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="login-box">
        <h1>🏟️ QuorumAI</h1>
        <p>Multi-LLM Aggregation Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    login_url = get_login_url()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "🐙  Sign in with GitHub",
            login_url,
            use_container_width=True,
        )
        st.caption("We only read your profile and email. No repo access.")

    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        st.error(
            "⚠️ GitHub OAuth is not configured. "
            "Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in your .env file."
        )
