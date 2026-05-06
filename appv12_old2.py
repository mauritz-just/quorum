"""
QuorumAI — Multi-LLM Aggregation Platform
=======================================
Architecture layers implemented:
  1. Frontend Layer          → Streamlit UI (prompt box, real-time feedback, results)
  2. Prompt Analyzer         → Heuristic quality/complexity scoring
  2.5 Prompt Refiner         → AI-powered rewrite (balanced / simplify / enhance)
  3. Resource Allocator      → Selects which models to call based on complexity
  3a Prompt Summariser       → ≤5-word topic summary for history panel
  3b Model Recommendation    → Suggests optimal model count + selection
  4. LLM Orchestration Layer → Parallel API calls to 4 models
  5. Aggregation Engine      → Merges outputs, synthesises consensus (via Gemini)
  6. Final Output            → Synthesised answer, confidence score, perspectives
  7. Session History         → Persistent query log with re-use

Run:  streamlit run appv11.py
"""

import streamlit as st
import requests
import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from db import init_db
from auth import require_auth, logout
from key_manager import (
    get_keys, build_models_dict, save_key, delete_key,
    toggle_key, test_key, PROVIDER_PRESETS, MAX_KEYS_PER_USER,
    count_active_keys, mask_key,
)

# ──────────────────────────────────────────────
# 1 · MODEL CONFIGURATION  (keys loaded from .env — never hardcoded)
# ──────────────────────────────────────────────
FALLBACK_MODELS = {
    "Gemini 2.5 Flash": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        "type": "gemini",
        "icon": "✦",
        "color": "#4285F4",
        "provider": "Google",
    },
    "Groq · Llama 3.3 70B": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "endpoint": "https://api.groq.com/openai/v1/chat/completions",
        "model_id": "llama-3.3-70b-versatile",
        "type": "openai_compat",
        "icon": "⚡",
        "color": "#F55036",
        "provider": "Groq",
    },
    "Mistral · Devstral Medium": {
        "api_key": os.getenv("MISTRAL_API_KEY", ""),
        "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "model_id": "devstral-medium-latest",
        "type": "openai_compat",
        "icon": "▲",
        "color": "#FF7000",
        "provider": "Mistral AI",
    },
    "Cerebras · Qwen3 235B": {
        "api_key": os.getenv("CEREBRAS_API_KEY", ""),
        "endpoint": "https://api.cerebras.ai/v1/chat/completions",
        "model_id": "qwen-3-235b-a22b-instruct-2507",
        "type": "openai_compat",
        "icon": "◆",
        "color": "#C04A3E",
        "provider": "Cerebras",
    },
}

# ──────────────────────────────────────────────
# 1b · AGGREGATOR FALLBACK via OpenRouter
# ──────────────────────────────────────────────
OPENROUTER_FALLBACK = {
    "api_key": os.getenv("OPENROUTER_API_KEY", ""),
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "model_id": "meta-llama/llama-3.3-70b-instruct:free",
    "name": "OpenRouter · Llama 3.3 70B (free)",
}

# MODELS is rebuilt after auth on every run (merged user keys + fallbacks)
MODELS = dict(FALLBACK_MODELS)

# ──────────────────────────────────────────────
# 2 · PROMPT ANALYZER  (Architecture Layer 2)
# ──────────────────────────────────────────────

_ACTION_VERBS = {
    "write", "analyze", "analyse", "generate", "create", "design", "build",
    "summarize", "summarise", "extract", "compare", "evaluate", "draft",
    "translate", "classify", "rewrite", "refactor", "debug", "explain",
    "describe", "list", "calculate", "compute", "review", "critique",
    "identify", "recommend", "plan", "outline", "propose",
}

_PREAMBLE_PATTERNS = [
    r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b",
    r"\bcould you (please )?", r"\bcan you (please )?", r"\bwould you (please )?",
    r"\bplease\b.*\bhelp", r"\bi was wondering\b", r"\bif you don'?t mind\b",
    r"\bi'?d like (you )?to\b", r"\bi want (you )?to\b",
    r"\bthank(s| you)\b", r"\bsorry (to bother|for)",
]

_OUTPUT_FORMAT_HINTS = [
    "json", "yaml", "xml", "csv", "markdown", "table", "bullet", "numbered list",
    "paragraph", "prose", "essay", "report", "summary", "outline", "code block",
    "in tags", "<thinking>", "step by step", "step-by-step",
    "word count", "words long", "sentences", "pages", "paragraphs",
    "format:", "structure:", "return only", "respond with only", "output:",
]

_CONTEXT_MARKERS = [
    "because", "so that", "in order to", "the goal is", "the purpose is",
    "the audience is", "for a", "for an", "intended for", "will be used for",
    "context:", "background:", "i'm a", "i am a", "we're", "we are",
    "as part of", "for my", "for our", "to help",
]

_UNCERTAINTY_SAFEGUARDS = [
    "if you don't know", "if you are unsure", "if uncertain",
    "say so", "admit", "acknowledge", "don't speculate", "do not speculate",
    "don't guess", "do not guess", "don't make up", "rather than guessing",
    "null if unsure", "say 'i don't know'", "express uncertainty",
    "flag any assumptions",
]

_COT_TRIGGERS = [
    "think step by step", "step-by-step", "step by step",
    "first", "then", "finally",
    "<thinking>", "reason through", "walk through your reasoning",
    "show your work", "explain your reasoning",
]

_NEGATIVE_FRAMING = [
    r"\bdon'?t\b", r"\bdo not\b", r"\bnever\b", r"\bno (bullet|markdown|emoji|preamble)",
    r"\bavoid\b", r"\bexclude\b", r"\bwithout\b",
]

_VAGUE_TERMS = {
    "thing", "stuff", "something", "anything", "good", "nice", "better",
    "some", "a lot", "etc", "and so on", "various", "relevant",
}


def _score_direction(text):
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    first_words = tokens[:4]
    has_action_verb_early = any(w in _ACTION_VERBS for w in first_words)
    preamble_hits = [p for p in _PREAMBLE_PATTERNS if re.search(p, text.lower())]
    preamble_count = len(preamble_hits)
    score = 0
    if has_action_verb_early:
        score += 60
    if preamble_count == 0:
        score += 40
    elif preamble_count == 1:
        score += 15
    evidence = []
    if has_action_verb_early:
        evidence.append(f"Opens with action verb: '{first_words[0] if first_words else ''}'")
    else:
        evidence.append("No clear action verb at the start")
    if preamble_count > 0:
        evidence.append(f"{preamble_count} preamble/filler phrase(s) detected")
    tip = None
    if not has_action_verb_early:
        tip = "Lead with a direct action verb (Analyze / Write / Generate / Extract)."
    elif preamble_count > 0:
        tip = "Drop the conversational filler. Jump straight to the request."
    return {"score": min(score, 100), "evidence": evidence, "tip": tip}


def _score_output_definition(text):
    lower = text.lower()
    hits = [h for h in _OUTPUT_FORMAT_HINTS if h in lower]
    has_structure = any(h in lower for h in ["json", "yaml", "xml", "table", "bullet", "markdown"])
    has_length = bool(re.search(r"\b(\d+)\s*(words?|sentences?|paragraphs?|bullets?|items?|lines?|pages?)\b", lower))
    has_structure_word = any(h in lower for h in ["format:", "structure:", "return only", "respond with only"])
    score = 0
    if has_structure:
        score += 40
    if has_length:
        score += 30
    if has_structure_word:
        score += 30
    if hits and score < 40:
        score = 40
    evidence = []
    if hits:
        evidence.append(f"Format hints: {', '.join(hits[:3])}")
    if has_length:
        evidence.append("Explicit length constraint present")
    if not hits and not has_length:
        evidence.append("No format, length, or structure specified")
    tip = None
    if score < 50:
        tip = "State the exact output shape — format (JSON / bullets / prose), length, and what sections it must contain."
    return {"score": min(score, 100), "evidence": evidence, "tip": tip}


def _score_context(text):
    lower = text.lower()
    has_why = any(m in lower for m in ["because", "so that", "in order to", "the goal"])
    has_audience = any(m in lower for m in ["audience", "for a", "for an", "intended for"])
    has_use_case = any(m in lower for m in ["will be used for", "as part of", "for my", "to help"])
    score = 0
    if has_why:
        score += 40
    if has_audience:
        score += 30
    if has_use_case:
        score += 30
    evidence = []
    if has_why:
        evidence.append("Explains the 'why' / goal")
    if has_audience:
        evidence.append("Audience identified")
    if has_use_case:
        evidence.append("Use case / downstream purpose stated")
    if score == 0:
        evidence.append("No context — model must guess audience, purpose, and goal")
    tip = None
    if score < 40:
        tip = "Add the 'why': who will read this, how the output will be used, what problem it solves."
    return {"score": min(score, 100), "evidence": evidence, "tip": tip}


def _score_specificity(text):
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", text)
    vague_hits = [v for v in _VAGUE_TERMS if v in tokens]
    proper_nouns = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    score = 20
    score += min(len(numbers) * 12, 30)
    score += min(len(proper_nouns) * 8, 20)
    score -= min(len(vague_hits) * 10, 30)
    if len(tokens) >= 25:
        score += 20
    score = max(0, min(score, 100))
    evidence = []
    if numbers:
        evidence.append(f"{len(numbers)} concrete number(s) / quantit(ies)")
    if proper_nouns:
        evidence.append(f"Named entities: {', '.join(proper_nouns[:3])}")
    if vague_hits:
        evidence.append(f"Vague terms to sharpen: {', '.join(vague_hits[:3])}")
    tip = None
    if vague_hits:
        tip = f"Replace vague words ({', '.join(vague_hits[:2])}) with concrete constraints."
    elif score < 50:
        tip = "Add concrete constraints: exact numbers, specific entity names, non-negotiable requirements."
    return {"score": score, "evidence": evidence, "tip": tip}


def _score_uncertainty(text):
    lower = text.lower()
    hits = [s for s in _UNCERTAINTY_SAFEGUARDS if s in lower]
    score = 100 if hits else 0
    evidence = []
    if hits:
        evidence.append(f"Anti-hallucination safeguard present: '{hits[0]}'")
    else:
        evidence.append("No uncertainty permission — model may confabulate if unsure")
    tip = None
    if score == 0:
        tip = "Add: \"If you are unsure or lack sufficient data, say so rather than speculating.\""
    return {"score": score, "evidence": evidence, "tip": tip}


def _score_cot(text, complexity):
    lower = text.lower()
    hits = [t for t in _COT_TRIGGERS if t in lower]
    has_cot = len(hits) > 0
    needs_cot = complexity >= 55
    if not needs_cot:
        return {"score": 100, "evidence": ["Task is simple — CoT not required"], "tip": None}
    if has_cot:
        return {"score": 100, "evidence": [f"CoT trigger present: '{hits[0]}'"], "tip": None}
    return {"score": 30, "evidence": ["Complex task without step-by-step reasoning trigger"],
            "tip": "Complex task — add 'Think step by step before answering' or structure it as First / Then / Finally."}


def _score_framing(text):
    negative_hits = sum(1 for p in _NEGATIVE_FRAMING if re.search(p, text.lower()))
    total_sentences = max(1, len(re.split(r"[.!?]+", text.strip())))
    ratio = negative_hits / total_sentences
    if negative_hits == 0:
        score = 100
    elif ratio < 0.3:
        score = 75
    elif ratio < 0.6:
        score = 45
    else:
        score = 20
    evidence = []
    if negative_hits == 0:
        evidence.append("Uses positive / architectural framing")
    else:
        evidence.append(f"{negative_hits} negative constraint(s) ('don't / no / avoid')")
    tip = None
    if score < 60:
        tip = "Replace 'don't do X' with 'do Y instead'. Negative constraints leave the output space underspecified."
    return {"score": score, "evidence": evidence, "tip": tip}


def analyse_prompt(text):
    words = text.split()
    word_count = len(words)
    sentence_count = max(1, len(re.split(r'[.!?]+', text.strip())) - 1 or 1)
    lower = text.lower()
    has_question = "?" in text
    has_code = any(kw in lower for kw in ["code", "function", "script", "program", "debug", "python", "javascript", "sql", "html", "css", "api"])
    has_creative = any(kw in lower for kw in ["write", "story", "poem", "essay", "blog", "creative", "imagine", "describe", "narrative"])
    has_analysis = any(kw in lower for kw in ["compare", "analyse", "analyze", "explain", "evaluate", "pros and cons", "difference", "versus", "vs"])
    complexity = 10
    complexity += min(word_count * 1.5, 40)
    complexity += 15 if has_code else 0
    complexity += 10 if has_analysis else 0
    complexity += 5 if has_creative else 0
    complexity += 10 if sentence_count > 3 else 0
    complexity = min(int(complexity), 100)
    if complexity < 30:
        level = "Simple"
    elif complexity < 60:
        level = "Moderate"
    else:
        level = "Complex"
    dims = {
        "Explicit Direction": _score_direction(text),
        "Defined Output": _score_output_definition(text),
        "Context & Motivation": _score_context(text),
        "Specificity": _score_specificity(text),
        "Uncertainty Safeguard": _score_uncertainty(text),
        "Chain of Thought": _score_cot(text, complexity),
        "Positive Framing": _score_framing(text),
    }
    weights = {"Explicit Direction": 0.18, "Defined Output": 0.18, "Context & Motivation": 0.16,
               "Specificity": 0.20, "Uncertainty Safeguard": 0.10, "Chain of Thought": 0.10, "Positive Framing": 0.08}
    quality = int(sum(dims[k]["score"] * w for k, w in weights.items()))
    if has_code:
        category = "💻 Code / Technical"
    elif has_creative:
        category = "✍️ Creative / Writing"
    elif has_analysis:
        category = "🔍 Analysis / Comparison"
    elif has_question:
        category = "❓ Question / Factual"
    else:
        category = "💬 General"
    ranked = sorted([(name, d) for name, d in dims.items() if d["tip"]], key=lambda kv: kv[1]["score"])
    suggestions = [f"**{name}** — {d['tip']}" for name, d in ranked[:4]]
    return {"word_count": word_count, "sentence_count": sentence_count, "complexity": complexity,
            "complexity_level": level, "quality": quality, "category": category,
            "suggestions": suggestions, "dimensions": dims}


# ──────────────────────────────────────────────
# 2.5 · PROMPT REFINER  (balanced / simplify / enhance)
# ──────────────────────────────────────────────
REFINER_SYSTEM_PROMPT = """You are a prompt-engineering expert. Your job is to REWRITE a user's prompt so it follows the seven principles of effective prompting:

1. **Explicit direction** — Lead with a clear action verb. Remove conversational filler.
2. **Defined output** — State the exact format, length, and sections.
3. **Context & motivation** — Add the "why": audience, purpose, usage.
4. **Specificity** — Replace vague terms with concrete numbers and names.
5. **Permission to express uncertainty** — Add "If unsure, say so rather than speculating."
6. **Chain of thought** (only if complex) — Add "Think step by step" or First/Then/Finally.
7. **Positive framing** — Replace "don't do X" with "do Y instead".

RULES:
- Preserve the user's original intent exactly.
- Fill gaps with reasonable defaults.
- Match the original language.

READABILITY RULES (critical):
- Keep every sentence under 25 words. If a sentence is longer, split it into two.
- Use one idea per sentence. Never chain multiple instructions with commas or semicolons into a single sentence.
- Put each distinct instruction or constraint on its own line or as a separate sentence.
- Avoid trailing run-on clauses at the end of the prompt — end cleanly.
- The rewritten prompt should be easy to scan and act on, not a dense paragraph.

OUTPUT FORMAT — respond with ONLY:

REWRITTEN_PROMPT:
<the improved prompt>

CHANGES_MADE:
- <one-line bullet per change>
"""


def refine_prompt(user_prompt, analysis, timeout=60, mode="balanced"):
    weak_dims = sorted(analysis["dimensions"].items(), key=lambda kv: kv[1]["score"])[:4]
    weakness_block = "\n".join(
        f"- {name} (scored {d['score']}/100): {d['tip']}" for name, d in weak_dims if d["tip"]
    ) or "- (prompt scored well on all dimensions; make minor polish only)"

    mode_instructions = {
        "balanced": "Rewrite the prompt to address those weaknesses while preserving the user's exact intent.",
        "simplify": ("Rewrite the prompt to be SHORTER and MORE DIRECT. Remove unnecessary detail, filler, and redundancy. "
                     "Keep only what's essential. Target: 50-70% of the original length. The goal is EFFICIENCY."),
        "enhance": ("Rewrite the prompt to be MORE DETAILED and THOROUGH. Add specificity: concrete numbers, named constraints, "
                    "explicit format requirements. Add context the user likely intended. Add chain-of-thought triggers if complex. "
                    "The goal is MAXIMUM QUALITY output."),
    }

    user_msg = f"""The user wrote this prompt:

---
{user_prompt}
---

The diagnostic analyzer flagged these weaknesses:
{weakness_block}

{mode_instructions.get(mode, mode_instructions["balanced"])} Follow the output format specified in the system prompt."""

    priority = ["Gemini 2.5 Flash", "Groq · Llama 3.3 70B", "Cerebras · Qwen3 235B"]
    candidates = [name for name in priority if name in MODELS]
    if not candidates:
        return {"refined": user_prompt, "changes": [], "error": "No refinement-capable model configured.", "refiner_model": None}

    last_error = None
    for refiner_model in candidates:
        cfg = MODELS[refiner_model]
        try:
            if cfg["type"] == "gemini":
                text = _refine_via_gemini(cfg, user_msg, timeout)
            else:
                text = _refine_via_openai_compat(cfg, user_msg, timeout)
            break
        except Exception as e:
            last_error = str(e)
            err_lower = last_error.lower()
            is_transient = any(s in err_lower for s in ["429", "rate limit", "rate_limit", "quota", "too many requests", "503", "service unavailable", "502", "bad gateway"])
            if not is_transient:
                return {"refined": user_prompt, "changes": [], "error": last_error, "refiner_model": refiner_model}
            continue
    else:
        return {"refined": user_prompt, "changes": [], "error": last_error or "All refiner models rate-limited.", "refiner_model": candidates[-1]}

    refined, changes = user_prompt, []
    rewritten_match = re.search(r"REWRITTEN_PROMPT:\s*(.+?)(?=\n\s*CHANGES_MADE:|\Z)", text, re.DOTALL | re.IGNORECASE)
    changes_match = re.search(r"CHANGES_MADE:\s*(.+?)\Z", text, re.DOTALL | re.IGNORECASE)
    if rewritten_match:
        refined = rewritten_match.group(1).strip()
        refined = re.sub(r"^[`\"']+|[`\"']+$", "", refined).strip()
    if changes_match:
        changes = [line.lstrip("-•*").strip() for line in changes_match.group(1).strip().splitlines() if line.strip() and line.lstrip("-•*").strip()]
    return {"refined": refined, "changes": changes, "error": None, "refiner_model": refiner_model}


def _refine_via_gemini(cfg, user_msg, timeout):
    url = f"{cfg['endpoint']}?key={cfg['api_key']}"
    combined = REFINER_SYSTEM_PROMPT + "\n\n" + user_msg
    payload = {"contents": [{"parts": [{"text": combined}]}], "generationConfig": {"maxOutputTokens": 1500, "temperature": 0.3, "thinkingConfig": {"thinkingBudget": 0}}}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini blocked the refinement request")
    cand = candidates[0]
    finish = cand.get("finishReason", "")
    if finish in ("SAFETY", "RECITATION"):
        raise RuntimeError(f"Gemini refused the refinement ({finish})")
    parts = cand.get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini returned no content")
    return parts[0].get("text", "")


def _refine_via_openai_compat(cfg, user_msg, timeout):
    headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
    payload = {"model": cfg["model_id"], "messages": [{"role": "system", "content": REFINER_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}], "max_tokens": 1500, "temperature": 0.3}
    resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────
# 3 · RESOURCE ALLOCATOR
# ──────────────────────────────────────────────
def allocate_models(analysis, selected_models):
    return selected_models


# ──────────────────────────────────────────────
# 3a · PROMPT SUMMARISER (max 5 words for history)
# ──────────────────────────────────────────────
_FILLER_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those",
    "and", "but", "or", "nor", "for", "so", "yet", "in", "on", "at",
    "to", "of", "with", "by", "from", "about", "into", "through",
    "please", "hi", "hello", "hey", "help", "want", "like", "just",
    "really", "very", "also", "some", "any", "all", "each", "every",
    "how", "what", "which", "who", "whom", "whose", "when", "where",
    "why", "if", "then", "than", "both", "either", "neither",
    "not", "no", "nor", "don't", "doesn't", "didn't", "won't",
    "can't", "couldn't", "shouldn't", "wouldn't", "isn't", "aren't",
    "me", "mine", "us", "ours", "myself", "ourselves",
}


def summarise_prompt(text, max_words=5):
    clean = re.sub(r"[\"'`\n\r]+", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    words = re.findall(r"[a-zA-ZäöüÄÖÜßéèê]+", clean)
    key_words = [w for w in words if w.lower() not in _FILLER_WORDS and len(w) > 2]
    if not key_words:
        return " ".join(text.split()[:max_words])
    result = key_words[:max_words]
    result[0] = result[0].capitalize()
    return " ".join(result)


# ──────────────────────────────────────────────
# 3b · MODEL RECOMMENDATION ENGINE
# ──────────────────────────────────────────────
MODEL_PRIORITY = ["Gemini 2.5 Flash", "Groq · Llama 3.3 70B", "Cerebras · Qwen3 235B", "Mistral · Devstral Medium"]


def recommend_models(analysis, available_models):
    complexity = analysis["complexity"]
    category = analysis.get("category", "")
    quality = analysis["quality"]
    n_available = len(available_models)

    if complexity < 30:
        count, reasoning, strategy = 1, "Simple prompt — one model is sufficient.", "speed"
    elif complexity < 50:
        count, reasoning, strategy = 2, "Moderate complexity — two models provide cross-verification.", "balanced"
    elif complexity < 75:
        count, reasoning, strategy = 3, "Complex prompt — three models ensure diverse perspectives.", "thorough"
    else:
        count, reasoning, strategy = min(4, n_available), "Highly complex — use all available models for maximum coverage.", "maximum"

    if "Code" in category or "Technical" in category:
        count = max(count, 2)
        reasoning += " Technical prompts benefit from cross-checking."
    if "Analysis" in category or "Comparison" in category:
        count = max(count, 2)
        reasoning += " Analytical tasks improve with multiple perspectives."
    if "Creative" in category:
        count = max(count, 2)
        reasoning += " Creative tasks benefit from stylistic diversity."
    if quality < 30 and count > 2:
        count = 2
        reasoning += " ⚠️ Prompt quality is low — consider refining first."

    count = min(count, n_available)
    prioritised = [m for m in MODEL_PRIORITY if m in available_models]
    remaining = [m for m in available_models if m not in prioritised]
    suggested = (prioritised + remaining)[:count]

    icon = {"speed": "⚡", "balanced": "⚖️", "thorough": "🔍", "maximum": "🧠"}[strategy]
    label = {"speed": "Speed Priority", "balanced": "Balanced", "thorough": "Thorough", "maximum": "Maximum Coverage"}[strategy]
    return {"count": count, "total_available": n_available, "suggested": suggested,
            "reasoning": reasoning, "strategy": strategy, "icon": icon, "label": label}


# ──────────────────────────────────────────────
# 4 · LLM ORCHESTRATION LAYER
# ──────────────────────────────────────────────
def _call_gemini(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    url = f"{config['endpoint']}?key={config['api_key']}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature, "thinkingConfig": {"thinkingBudget": 0}}}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise RuntimeError(f"Gemini blocked the prompt (reason: {feedback.get('blockReason', 'unknown')})")
    cand = candidates[0]
    finish_reason = cand.get("finishReason", "")
    if finish_reason == "SAFETY":
        raise RuntimeError("Gemini blocked the response due to content/safety policy")
    if finish_reason == "RECITATION":
        raise RuntimeError("Gemini blocked the response due to a recitation trigger")
    parts = cand.get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini returned no content (finish_reason: {finish_reason or 'unknown'})")
    text = parts[0].get("text", "")
    if finish_reason == "MAX_TOKENS":
        text += "\n\n⚠️ *Response truncated — increase 'Max response length' in Settings.*"
    return text


def _call_openai_compat(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    headers = {"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"}
    payload = {"model": config["model_id"], "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    resp = requests.post(config["endpoint"], headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    headers = {"x-api-key": config["api_key"], "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {"model": config["model_id"], "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(config["endpoint"], headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def call_model(name, prompt, max_tokens=1024, temperature=0.7, timeout=90):
    config = MODELS[name]
    start = time.time()
    try:
        if config["type"] == "gemini":
            text = _call_gemini(prompt, config, max_tokens, temperature, timeout)
        elif config["type"] == "anthropic":
            text = _call_anthropic(prompt, config, max_tokens, temperature, timeout)
        else:
            text = _call_openai_compat(prompt, config, max_tokens, temperature, timeout)
        return {"name": name, "response": text, "time": round(time.time() - start, 2), "words": len(text.split()), "error": None}
    except Exception as e:
        return {"name": name, "response": None, "time": round(time.time() - start, 2), "words": 0, "error": str(e)}


def orchestrate(prompt, model_names, max_tokens=1024, temperature=0.7, timeout=90):
    results = []
    with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
        futures = {pool.submit(call_model, name, prompt, max_tokens, temperature, timeout): name for name in model_names}
        for future in as_completed(futures):
            results.append(future.result())
    order = {name: i for i, name in enumerate(model_names)}
    results.sort(key=lambda r: order.get(r["name"], 99))
    return results


# ──────────────────────────────────────────────
# 4b · ERROR SAFETY NET
# ──────────────────────────────────────────────
def interpret_error(raw_error, model_name=""):
    if not raw_error:
        return {"icon": "❓", "title": "Unknown error", "message": "The model returned no response and no error.", "hint": "Try running the prompt again.", "raw": ""}
    err = str(raw_error)
    err_lower = err.lower()
    if any(s in err_lower for s in ["429", "rate limit", "rate_limit", "quota", "too many requests"]):
        return {"icon": "🚦", "title": "Rate limit reached", "message": f"{model_name} is temporarily throttling requests.", "hint": "Wait 30–60 seconds and try again.", "raw": err}
    if any(s in err_lower for s in ["401", "unauthorized", "invalid api key", "invalid_api_key", "authentication"]):
        return {"icon": "🔑", "title": "Authentication failed", "message": f"The API key for {model_name} was rejected.", "hint": "Check the provider's dashboard.", "raw": err}
    if any(s in err_lower for s in ["403", "forbidden", "access denied", "region"]):
        return {"icon": "🚫", "title": "Access denied", "message": f"{model_name} refused the request — possibly a regional block.", "hint": "Some free tiers are geo-restricted.", "raw": err}
    if any(s in err_lower for s in ["402", "insufficient", "payment required", "balance", "credits"]):
        return {"icon": "💳", "title": "Insufficient credits", "message": f"{model_name} has no remaining credits.", "hint": "Top up or switch to another model.", "raw": err}
    if any(s in err_lower for s in ["model not found", "model_not_found", "does not exist", "deprecated"]):
        return {"icon": "🗂️", "title": "Model unavailable", "message": f"The model ID for {model_name} is no longer valid.", "hint": "Update the model_id in config.", "raw": err}
    if any(s in err_lower for s in ["context length", "context_length", "token limit", "too long"]):
        return {"icon": "📏", "title": "Prompt too long", "message": f"Prompt exceeded {model_name}'s context window.", "hint": "Shorten the prompt.", "raw": err}
    if any(s in err_lower for s in ["timeout", "timed out", "read timed out"]):
        return {"icon": "⏱️", "title": "Request timed out", "message": f"{model_name} did not respond in time.", "hint": "Increase the timeout or try again.", "raw": err}
    if any(s in err_lower for s in ["500", "502", "503", "504", "internal server", "bad gateway", "service unavailable"]):
        return {"icon": "🛠️", "title": "Provider outage", "message": f"{model_name}'s servers returned an error.", "hint": "Wait a minute and retry.", "raw": err}
    if any(s in err_lower for s in ["content policy", "content_policy", "safety", "blocked", "moderation", "harmful"]):
        return {"icon": "🛡️", "title": "Content policy block", "message": f"{model_name} refused due to content policy.", "hint": "Reword the prompt or try another model.", "raw": err}
    if any(s in err_lower for s in ["connection", "network", "dns", "name resolution", "ssl", "max retries"]):
        return {"icon": "📡", "title": "Network problem", "message": f"Could not reach {model_name}'s servers.", "hint": "Check your internet connection.", "raw": err}
    if any(s in err_lower for s in ["400", "bad request", "invalid_request", "invalid request"]):
        return {"icon": "⚠️", "title": "Malformed request", "message": f"{model_name} rejected the request format.", "hint": "Check the provider's docs.", "raw": err}
    return {"icon": "❌", "title": "Unexpected error", "message": f"{model_name} returned an unclassified error.", "hint": "See raw error below.", "raw": err}


# ──────────────────────────────────────────────
# 5 · AGGREGATION ENGINE
# ──────────────────────────────────────────────
AGGREGATION_PROMPT = """You are an expert aggregation engine. You have received responses from {count} different AI models to the same user prompt.

USER PROMPT:
\"\"\"{user_prompt}\"\"\"

MODEL RESPONSES:
{responses_block}

YOUR TASK — produce a structured synthesis:

1. **Consensus Summary**: What do most or all models agree on? Write a clear, merged answer that captures the shared knowledge. Remove redundancies.

2. **Unique Insights**: List any valuable points that only one or two models mentioned and that add value.

3. **Disagreements**: Note any areas where models contradicted each other. State each position fairly.

4. **Confidence Score**: Rate from 1-10 how confident the combined answer is (10 = perfect agreement on a factual matter, 1 = wild disagreement).

5. **Recommended Answer**: Write the single best answer you would give the user, drawing on all models.

Respond in clean Markdown."""


def aggregate_responses(user_prompt, results):
    valid = [r for r in results if r["response"]]
    if len(valid) < 2:
        return None

    responses_block = ""
    for r in valid:
        responses_block += f"\n--- {r['name']} ---\n{r['response']}\n"

    meta_prompt = AGGREGATION_PROMPT.format(count=len(valid), user_prompt=user_prompt, responses_block=responses_block)

    # ── Attempt 1: Gemini (first available Gemini model) ──
    gemini_error = None
    _gemini_entry = next(
        ((n, c) for n, c in MODELS.items() if c.get("type") == "gemini" and c.get("api_key")), None
    )
    if not _gemini_entry:
        gemini_error = RuntimeError("No Gemini model configured")
    else:
        _gemini_name, gemini_cfg = _gemini_entry
        url = f"{gemini_cfg['endpoint']}?key={gemini_cfg['api_key']}"
        payload = {"contents": [{"parts": [{"text": meta_prompt}]}], "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3, "thinkingConfig": {"thinkingBudget": 0}}}
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")
            text = candidates[0]["content"]["parts"][0]["text"]
            return text + f"\n\n---\n*Aggregated by {_gemini_name}*"
        except Exception as e:
            gemini_error = e

    # ── Attempt 2: OpenRouter fallback ──
    if not OPENROUTER_FALLBACK.get("api_key"):
        return f"⚠️ Gemini aggregation failed and no OpenRouter key configured.\n\nGemini error: {gemini_error}"

    try:
        headers = {"Authorization": f"Bearer {OPENROUTER_FALLBACK['api_key']}", "Content-Type": "application/json"}
        payload = {"model": OPENROUTER_FALLBACK["model_id"], "messages": [{"role": "user", "content": meta_prompt}], "max_tokens": 2048, "temperature": 0.3}
        resp = requests.post(OPENROUTER_FALLBACK["endpoint"], headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text + f"\n\n---\n*⚠️ Aggregated by {OPENROUTER_FALLBACK['name']} (fallback)*"
    except Exception as e:
        or_error = e

    # ── Attempt 3: first OpenAI-compat model fallback ──
    _compat_entry = next(
        ((n, c) for n, c in MODELS.items() if c.get("type") == "openai_compat" and c.get("api_key") and c.get("model_id")), None
    )
    if not _compat_entry:
        return f"⚠️ Aggregation failed on all models.\n\n**Gemini:** {gemini_error}\n\n**OpenRouter:** {or_error}"
    try:
        _compat_name, groq_cfg = _compat_entry
        headers = {"Authorization": f"Bearer {groq_cfg['api_key']}", "Content-Type": "application/json"}
        payload = {"model": groq_cfg["model_id"], "messages": [{"role": "user", "content": meta_prompt}], "max_tokens": 2048, "temperature": 0.3}
        resp = requests.post(groq_cfg["endpoint"], headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return text + f"\n\n---\n*⚠️ Aggregated by {_compat_name} (fallback)*"
    except Exception as groq_err:
        return f"⚠️ Aggregation failed on all models.\n\n**Gemini:** {gemini_error}\n\n**OpenRouter:** {or_error}\n\n**Fallback:** {groq_err}"


# ──────────────────────────────────────────────
# 6 · FRONTEND LAYER — Streamlit UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="QuorumAI · Model Arena & Aggregator", page_icon="🏟️", layout="wide", initial_sidebar_state="expanded")

# ── Auth & DB ──
init_db()
user = require_auth()
if not user:
    st.stop()

# ── Build MODELS from user keys + fallbacks ──
_user_id = user["id"]
if "user_keys" not in st.session_state:
    st.session_state.user_keys = get_keys(_user_id)

MODELS = build_models_dict(st.session_state.user_keys, FALLBACK_MODELS)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .top-bar { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1.5rem; display: flex; align-items: baseline; gap: 1rem; }
    .top-bar h1 { color: #fff; font-size: 1.6rem; margin: 0; font-weight: 700; line-height: 1.2; }
    .top-bar span { color: #94a3b8; font-size: 0.95rem; line-height: 1.2; }
    .model-card { border: 1px solid #2a2a3a; border-radius: 10px; padding: 1rem; background: #0e1117; margin-bottom: 0.5rem; }
    .model-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1e1e2e; }
    .model-name { font-weight: 700; font-size: 0.95rem; }
    .model-meta { font-size: 0.78rem; color: #64748b; }
    .metric-pill { display: inline-block; background: #1e293b; padding: 0.25rem 0.7rem; border-radius: 20px; font-size: 0.78rem; color: #94a3b8; margin-right: 0.4rem; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="top-bar"><h1>🏟️ QuorumAI — Model Arena & Aggregator</h1><span>Multi-LLM Aggregation Platform · Compare · Synthesise · Decide</span></div>', unsafe_allow_html=True)

# ── Sidebar ──
with st.sidebar:
    # ── User info ──
    _ava_col, _info_col = st.columns([1, 3])
    with _ava_col:
        if user.get("avatar_url"):
            st.image(user["avatar_url"], width=40)
        else:
            st.markdown("👤")
    with _info_col:
        st.markdown(f"**{user.get('display_name', 'User')}**")
        if user.get("email"):
            st.caption(user["email"])
    if st.button("🚪 Sign out", use_container_width=True, key="signout_btn"):
        logout()
        st.rerun()
    st.markdown("---")

    mode = st.radio("Mode", ["🏟️ Arena (side-by-side)", "🧠 Aggregator (synthesised)"], index=0)
    st.markdown("---")
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    if st.button("✕ Close Settings" if st.session_state.show_settings else "⚙️ Settings", use_container_width=True, key="settings_toggle"):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()

    for key, default in [("s_max_tokens", 1024), ("s_temperature", 0.7), ("s_timeout", 240), ("s_show_analysis", True)]:
        if key not in st.session_state:
            st.session_state[key] = default

    def _save(src, dst):
        st.session_state[dst] = st.session_state[src]

    if st.session_state.show_settings:
        st.markdown("### 🛠️ Generation Settings")
        st.slider("Max response length (tokens)", 100, 5000, st.session_state.s_max_tokens, 100, key="_w_mt", on_change=lambda: _save("_w_mt", "s_max_tokens"))
        st.slider("Temperature (creativity)", 0.0, 1.5, st.session_state.s_temperature, 0.1, key="_w_temp", on_change=lambda: _save("_w_temp", "s_temperature"))
        st.slider("Timeout per model (seconds)", 30, 600, st.session_state.s_timeout, 30, key="_w_to", on_change=lambda: _save("_w_to", "s_timeout"))
        st.toggle("Show Prompt Analysis", value=st.session_state.s_show_analysis, key="_w_sa", on_change=lambda: _save("_w_sa", "s_show_analysis"))
        st.markdown("---")

    max_tokens = st.session_state.s_max_tokens
    temperature = st.session_state.s_temperature
    timeout_sec = st.session_state.s_timeout
    show_prompt_analysis = st.session_state.s_show_analysis

    st.markdown("### 🤖 Models")
    st.caption("Toggle which models receive your prompt (max 5):")
    selected_models = []
    for name, cfg in MODELS.items():
        label = f"{cfg['icon']} {name}"
        if st.checkbox(label, value=True, key=f"chk_{name}"):
            selected_models.append(name)
    if not selected_models:
        st.error("Select at least 1 model.")
    elif len(selected_models) > 5:
        st.warning("Max 5 models — only the first 5 will be used.")
        selected_models = selected_models[:5]
    else:
        st.caption(f"{len(selected_models)} of {len(MODELS)} models active")

    st.markdown("---")
    if "s_show_value_prop" not in st.session_state:
        st.session_state.s_show_value_prop = False
    st.session_state.s_show_value_prop = st.toggle("💡 Show Value Proposition", value=st.session_state.s_show_value_prop)
    if st.session_state.s_show_value_prop:
        st.markdown("Our tool earns a spot in your daily workflow for one of three reasons. **Cross-verification** for students and researchers. **Best answer, guaranteed** for professionals. **Cost efficiency** for companies with multiple AI subscriptions. We run all models **in parallel**, add a **synthesis layer** with confidence scores, make **disagreement transparent**, and do it all in **one prompt box**.")
    # ── API Key Management ──
    st.markdown("---")
    st.markdown("### 🔑 My API Keys")
    _user_keys = st.session_state.user_keys
    _active_count = sum(1 for k in _user_keys if k["is_active"])
    st.caption(f"{_active_count} of {MAX_KEYS_PER_USER} slots used")

    for _k in _user_keys:
        _preset = PROVIDER_PRESETS.get(_k["provider_name"], {})
        _icon = _preset.get("icon", "🔑")
        _kc1, _kc2, _kc3 = st.columns([4, 1, 1])
        with _kc1:
            _status = "✅" if _k["is_active"] else "⏸"
            st.markdown(f"{_icon} **{_k['display_name'] or _k['provider_name']}** {_status}")
            st.caption(f"`{_k['masked_key']}` · {_k['model_id']}")
        with _kc2:
            _tog_label = "Pause" if _k["is_active"] else "Enable"
            if st.button("⏸" if _k["is_active"] else "▶", key=f"tog_{_k['id']}", help=_tog_label):
                toggle_key(_user_id, _k["id"])
                st.session_state.user_keys = get_keys(_user_id)
                st.rerun()
        with _kc3:
            if st.button("🗑", key=f"del_{_k['id']}", help="Delete key"):
                delete_key(_user_id, _k["id"])
                st.session_state.user_keys = get_keys(_user_id)
                st.rerun()

    if _active_count < MAX_KEYS_PER_USER:
        with st.expander("➕ Add API Key"):
            _provider = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()), key="new_key_provider")
            _preset_cfg = PROVIDER_PRESETS[_provider]
            if _preset_cfg["models"]:
                _model_id = st.selectbox("Model", _preset_cfg["models"], key="new_key_model")
            else:
                _model_id = st.text_input("Model ID", key="new_key_model_custom")
            _endpoint = _preset_cfg["endpoint"]
            if not _endpoint:
                _endpoint = st.text_input("Endpoint URL", key="new_key_endpoint")
            _display_name = st.text_input("Display name (optional)", key="new_key_display")
            _api_key_input = st.text_input("API Key", type="password", key="new_key_value")
            if st.button("🧪 Test & Save", key="save_new_key", use_container_width=True):
                if not _api_key_input:
                    st.error("Enter an API key.")
                elif not _endpoint:
                    st.error("Enter an endpoint URL.")
                else:
                    with st.spinner("Testing key…"):
                        _ok, _msg = test_key(_provider, _api_key_input, _model_id, _endpoint, _preset_cfg["api_type"])
                    if _ok:
                        try:
                            save_key(_user_id, _provider, _display_name or _provider, _api_key_input, _model_id, _endpoint, _preset_cfg["api_type"])
                            st.session_state.user_keys = get_keys(_user_id)
                            st.success("✅ Key saved!")
                            st.rerun()
                        except ValueError as _e:
                            st.error(str(_e))
                    else:
                        st.error(f"❌ Key test failed: {_msg}")
    else:
        st.info(f"Maximum of {MAX_KEYS_PER_USER} keys reached. Delete or pause one to add another.")

    # ── Prompt History ──
    st.markdown("---")
    st.markdown("### 💬 History")
    if not st.session_state.get("query_history"):
        st.caption("No prompts yet.")
    else:
        st.caption(f"{len(st.session_state.query_history)} prompt(s) this session")
        for _hi, _entry in enumerate(reversed(st.session_state.query_history)):
            _hidx = len(st.session_state.query_history) - _hi
            _mode_icon = "🏟️" if _entry["mode"] == "Arena" else ("🧠" if _entry["mode"] == "Aggregator" else "✨")
            _quality = f" · {_entry['quality']}/100" if _entry.get("quality") else ""
            _label = f"#{_hidx} {_mode_icon} {_entry.get('summary', 'Query')} · {_entry['timestamp']}{_quality}"
            if st.button(_label, key=f"sidebar_reuse_{_hidx}", use_container_width=True):
                st.session_state._prompt_to_apply = _entry["full_prompt"]
                st.session_state.last_analysis = None
                st.session_state.refinement_history = []
                st.rerun()
        if st.button("🗑️ Clear history", use_container_width=True, key="sidebar_clear_history"):
            st.session_state.query_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("### 👷 Architects")
    st.caption("1. Jan Schützenmeister\n\n2. Mauritz Horst Friedrich Just")

# ── Main Area ──
for key, default in [
    ("working_prompt", ""),
    ("last_analysis", None),
    ("refinement_history", []),
    ("broadcast_now", False),
    ("query_history", []),
    ("_pending_refined", None),
    ("_pending_analysis", None),
    ("_prompt_to_apply", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Apply refined/reused prompt BEFORE the text_area widget is instantiated
if st.session_state._prompt_to_apply is not None:
    st.session_state.working_prompt = st.session_state._prompt_to_apply
    st.session_state.prompt_input = st.session_state._prompt_to_apply
    st.session_state._prompt_to_apply = None

user_prompt = st.text_area(
    "Enter your prompt — we'll diagnose it before broadcasting:",
    value=st.session_state.working_prompt,
    height=140,
    key="prompt_input",
    placeholder="e.g.  Explain the difference between REST and GraphQL APIs with code examples…"
)

st.session_state.working_prompt = user_prompt

# ── Model count selector (forced in Aggregator mode) ──
if "Aggregator" in mode and user_prompt.strip() and selected_models:
    quick_analysis = analyse_prompt(user_prompt)
    rec = recommend_models(quick_analysis, selected_models)
    st.markdown(f'<div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border: 2px solid #0f3460; border-radius: 12px; padding: 1rem 1.5rem; margin: 0.5rem 0 1rem 0;"><span style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0;">{rec["icon"]} Resource Allocation: {rec["label"]}</span><br><span style="font-size: 0.85rem; color: #94a3b8;">{rec["reasoning"][:150]}</span></div>', unsafe_allow_html=True)

    alloc_col1, alloc_col2 = st.columns([1, 2])
    with alloc_col1:
        model_count_choice = st.select_slider("Models to use", options=list(range(1, len(selected_models) + 1)), value=rec["count"], key="model_count_slider",
                                              help="Slide to override the recommendation. More = better coverage but slower.")
    with alloc_col2:
        prioritised = [m for m in MODEL_PRIORITY if m in selected_models]
        remaining = [m for m in selected_models if m not in prioritised]
        ordered = prioritised + remaining
        active = ordered[:model_count_choice]
        inactive = ordered[model_count_choice:]
        active_str = "  ".join(f"**{MODELS[m]['icon']} {m}**" for m in active)
        st.markdown(f"🟢 Active: {active_str}")
        if inactive:
            inactive_str = "  ".join(f"~~{MODELS[m]['icon']} {m}~~" for m in inactive)
            st.markdown(f"⚪ Skipped: {inactive_str}")
else:
    model_count_choice = len(selected_models)

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    analyze_clicked = st.button("🔎 Analyze prompt", use_container_width=True, disabled=not user_prompt.strip())
with col_b:
    broadcast_clicked = st.button("📡 Broadcast to all models", type="primary", use_container_width=True, disabled=not user_prompt.strip() or not selected_models)
with col_c:
    if not selected_models:
        st.warning("Select at least one model in the sidebar.")
    elif "Aggregator" in mode:
        st.caption(f"📡 {model_count_choice} model(s) will be called  ·  slide above to adjust")
    else:
        st.caption(f"{len(selected_models)} model(s) selected  ·  mode: Arena")

# ── STAGE 1: Analyze & Refine ──
if analyze_clicked and user_prompt.strip():
    st.session_state.last_analysis = analyse_prompt(user_prompt)
    st.session_state._pending_refined = None
    st.session_state._pending_analysis = None
    if not st.session_state.refinement_history:
        st.session_state.refinement_history.append({"version": "v1 (original)", "prompt": user_prompt, "quality": st.session_state.last_analysis["quality"], "changes": []})

analysis = st.session_state.last_analysis

if analysis and show_prompt_analysis:
    st.markdown("---")
    st.markdown("### 🔎 Prompt Diagnostic")
    m1, m2, m3 = st.columns(3)
    m1.metric("Complexity", f"{analysis['complexity']}/100", analysis["complexity_level"])
    m2.metric("Overall Quality", f"{analysis['quality']}/100")
    if len(st.session_state.refinement_history) > 1:
        delta = analysis["quality"] - st.session_state.refinement_history[0]["quality"]
        m3.metric("Δ vs. v1", f"{delta:+d} pts")
    else:
        m3.metric("Iteration", f"v{len(st.session_state.refinement_history)}")

    st.markdown("#### 📐 Seven-dimension breakdown")
    dim_items = list(analysis["dimensions"].items())
    left, right = st.columns(2)
    for i, (name, d) in enumerate(dim_items):
        col = left if i % 2 == 0 else right
        with col:
            score = d["score"]
            emoji = "🟢" if score >= 75 else ("🟡" if score >= 45 else "🔴")
            with st.expander(f"{emoji} **{name}** — {score}/100", expanded=False):
                for ev in d["evidence"]:
                    st.markdown(f"• {ev}")
                if d["tip"]:
                    st.info(f"💡 {d['tip']}")
                else:
                    st.success("✓ No action needed.")

    if analysis["suggestions"]:
        st.markdown("#### 🎯 Top priorities to improve this prompt")
        for s in analysis["suggestions"]:
            st.markdown(f"- {s}")

    if "Aggregator" in mode:
        st.markdown("#### 🤖 Model Recommendation")
        rec = recommend_models(analysis, selected_models)
        rc_left, rc_right = st.columns([1, 2])
        with rc_left:
            st.metric(f"{rec['icon']} Strategy", rec["label"], f"{rec['count']} of {rec['total_available']} models")
        with rc_right:
            st.markdown(f"**Why:** {rec['reasoning']}")
            suggested_str = ", ".join(f"{MODELS[m]['icon']} {m}" for m in rec["suggested"])
            st.markdown(f"**Suggested:** {suggested_str}")

    # ── Refinement ──
    st.markdown("---")
    st.markdown("### ✨ Refine your prompt")
    st.caption("Refine using AI, or edit manually and re-analyse.")
    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        refine_clicked = st.button("✨ Balanced Refine", use_container_width=True)
    with btn2:
        simplify_clicked = st.button("⚡ Simplify", use_container_width=True)
    with btn3:
        enhance_clicked = st.button("🔬 Enhance", use_container_width=True)

    refine_mode = None
    if refine_clicked:
        refine_mode = "balanced"
    elif simplify_clicked:
        refine_mode = "simplify"
    elif enhance_clicked:
        refine_mode = "enhance"

    if refine_mode:
        spinner_msgs = {"balanced": "Rewriting your prompt…", "simplify": "⚡ Simplifying…", "enhance": "🔬 Enhancing…"}
        with st.spinner(spinner_msgs[refine_mode]):
            refinement = refine_prompt(user_prompt, analysis, timeout=timeout_sec, mode=refine_mode)
        if refinement["error"]:
            err_info = interpret_error(refinement["error"], refinement.get("refiner_model", "Refiner"))
            st.error(f"{err_info['icon']} **{err_info['title']}** — {err_info['message']}")
            st.caption(f"💡 {err_info['hint']}")
        else:
            refined_text = refinement["refined"]
            new_analysis = analyse_prompt(refined_text)
            mode_label = {"balanced": "AI refined", "simplify": "simplified", "enhance": "enhanced"}[refine_mode]
            st.session_state.refinement_history.append({"version": f"v{len(st.session_state.refinement_history) + 1} ({mode_label})", "prompt": refined_text, "quality": new_analysis["quality"], "changes": refinement["changes"]})
            # Store the refined result for display — don't auto-apply
            st.session_state._pending_refined = refined_text
            st.session_state._pending_analysis = new_analysis
            st.session_state._pending_old_quality = analysis['quality']
            st.session_state._pending_refiner = refinement.get('refiner_model', 'AI')

    # ── Show pending refined prompt with Use Prompt button ──
    if st.session_state.get("_pending_refined"):
        refined_text = st.session_state._pending_refined
        pending_analysis = st.session_state._pending_analysis
        old_q = st.session_state._pending_old_quality
        refiner_name = st.session_state._pending_refiner
        st.success(f"Refined with {refiner_name} · quality {old_q} → {pending_analysis['quality']}")
        st.code(refined_text, language="text")
        use_col, dismiss_col, _ = st.columns([1, 1, 3])
        with use_col:
            if st.button("✅ Use This Prompt", type="primary", use_container_width=True, key="use_refined"):

                # 1. Store original prompt in history (before overwriting)
                if user_prompt.strip():
                    original_analysis = analyse_prompt(user_prompt)

                    if not st.session_state.query_history or st.session_state.query_history[-1]["full_prompt"] != user_prompt:
                        st.session_state.query_history.append({
                            "timestamp": time.strftime("%H:%M:%S"),
                            "summary": summarise_prompt(user_prompt),
                            "prompt": user_prompt[:120] + ("…" if len(user_prompt) > 120 else ""),
                            "full_prompt": user_prompt,
                            "mode": "Prompt Refinement",
                            "quality": original_analysis["quality"],
                            "complexity": original_analysis["complexity_level"],
                            "models_used": [],
                            "successful": [],
                            "failed": [],
                            "times": {},
                            "responses": {},
                            "synthesis": None,
                        })

                # 2. Apply refined prompt
                st.session_state._prompt_to_apply = refined_text
                st.session_state.last_analysis = pending_analysis
                st.session_state._pending_refined = None
                st.session_state._pending_analysis = None

                st.rerun()
         
        with dismiss_col:
            if st.button("✕ Dismiss", use_container_width=True, key="dismiss_refined"):
                st.session_state._pending_refined = None
                st.session_state._pending_analysis = None
                st.rerun()

    if len(st.session_state.refinement_history) > 1:
        with st.expander(f"📜 Refinement history ({len(st.session_state.refinement_history)} versions)", expanded=False):
            for hist_i, entry in enumerate(reversed(st.session_state.refinement_history)):
                st.markdown(f"**{entry['version']}** — quality: {entry['quality']}/100")
                for ch in entry.get("changes", []):
                    st.markdown(f"  • {ch}")
                st.code(entry["prompt"], language="text")
                if entry["prompt"] != user_prompt:
                    if st.button(f"✅ Use This Prompt", key=f"use_hist_{hist_i}"):
                        st.session_state._prompt_to_apply = entry["prompt"]
                        st.session_state.last_analysis = None
                        st.session_state._pending_refined = None
                        st.rerun()

    st.markdown("---")
    st.info("👆 When you're happy, click **📡 Broadcast to all models** to send it out.")

# ── STAGE 2: Broadcast ──
if broadcast_clicked and user_prompt.strip():
    analysis = analyse_prompt(user_prompt)
    st.markdown("---")
    st.markdown("### 📡 Broadcasting")
    st.caption(f"Final prompt quality: **{analysis['quality']}/100** · complexity: **{analysis['complexity_level']}** · {model_count_choice} model(s)")

    if "Aggregator" in mode:
        rec = recommend_models(analysis, selected_models)
        if rec["count"] < len(selected_models):
            st.info(f"{rec['icon']} **Recommendation:** {rec['label']} — {rec['count']} model(s) optimal. You're using {model_count_choice}.")
        prioritised = [m for m in MODEL_PRIORITY if m in selected_models]
        remaining = [m for m in selected_models if m not in prioritised]
        ordered = prioritised + remaining
        models_to_call = ordered[:model_count_choice]
    else:
        models_to_call = allocate_models(analysis, selected_models)

    # ── Layer 4: LLM Orchestration with progress bar ──
    n_models = len(models_to_call)
    progress_bar = st.progress(0, text=f"📡 Calling {n_models} model(s) in parallel…")
    status_area = st.empty()

    results = []
    with ThreadPoolExecutor(max_workers=n_models) as pool:
        futures = {
            pool.submit(call_model, name, user_prompt, max_tokens, temperature, timeout_sec): name
            for name in models_to_call
        }
        completed_names = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_names.append(result["name"])
            pct = len(completed_names) / n_models
            cfg = MODELS[result["name"]]
            status_icon = "✅" if result["response"] else "❌"
            progress_bar.progress(pct, text=f"{status_icon} {cfg['icon']} {result['name']} done ({result['time']}s)  ·  {len(completed_names)}/{n_models} complete")

    # Re-sort to original order
    order = {name: i for i, name in enumerate(models_to_call)}
    results.sort(key=lambda r: order.get(r["name"], 99))

    progress_bar.progress(1.0, text=f"✅ All {n_models} model(s) finished")
    time.sleep(0.5)
    progress_bar.empty()
    status_area.empty()

    is_arena = "Arena" in mode
    synthesis = None

    if is_arena:
        st.markdown("#### 🏟️ Arena — Side-by-Side Comparison")
        successful = [r for r in results if r["response"]]
        if successful:
            fastest = min(successful, key=lambda r: r["time"])
            longest = max(successful, key=lambda r: r["words"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("⚡ Fastest", fastest["name"], f"{fastest['time']}s")
            mc2.metric("📝 Most Detailed", longest["name"], f"{longest['words']} words")
            mc3.metric("✅ Responses", f"{len(successful)}/{len(results)}")
        cols = st.columns(len(results))
        for i, result in enumerate(results):
            cfg = MODELS[result["name"]]
            with cols[i]:
                st.markdown(f'<div class="model-card"><div class="model-header"><span class="model-name" style="color:{cfg["color"]}">{cfg["icon"]} {result["name"]}</span><span class="model-meta">{result["time"]}s</span></div></div>', unsafe_allow_html=True)
                if result["error"]:
                    err_info = interpret_error(result["error"], result["name"])
                    st.error(f"{err_info['icon']} **{err_info['title']}**\n\n{err_info['message']}")
                    st.caption(f"💡 {err_info['hint']}")
                else:
                    st.markdown(f'<span class="metric-pill">⏱ {result["time"]}s</span><span class="metric-pill">📝 {result["words"]} words</span>', unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(result["response"])
    else:
        st.markdown("#### 🧠 Aggregator — Synthesised Output")
        with st.expander("📋 Individual Model Responses", expanded=False):
            for result in results:
                cfg = MODELS[result["name"]]
                if result["response"]:
                    st.markdown(f"**{cfg['icon']} {result['name']}** — *{result['time']}s · {result['words']} words*")
                    st.markdown(result["response"])
                    st.markdown("---")
                else:
                    err_info = interpret_error(result["error"], result["name"])
                    st.warning(f"{err_info['icon']} **{cfg['icon']} {result['name']} — {err_info['title']}**\n\n{err_info['message']}")

        with st.spinner("🧠 Aggregation Engine: merging outputs…"):
            synthesis = aggregate_responses(user_prompt, results)

        if synthesis:
            st.markdown("### 📌 Final Synthesised Answer")
            st.markdown(synthesis)
            st.markdown("---")
            src_names = [r["name"] for r in results if r["response"]]
            st.caption(f"Sources: {', '.join(src_names)}")
            st.download_button("📥 Download Result", data=synthesis, file_name="quorum_result.md", mime="text/markdown")
        else:
            st.warning("Not enough successful responses to aggregate (need at least 2).")

    # ── Save to history ──
    history_entry = {
        "timestamp": time.strftime("%H:%M:%S"), "summary": summarise_prompt(user_prompt),
        "prompt": user_prompt[:120] + ("…" if len(user_prompt) > 120 else ""), "full_prompt": user_prompt,
        "mode": "Arena" if is_arena else "Aggregator", "quality": analysis["quality"],
        "complexity": analysis["complexity_level"], "models_used": [r["name"] for r in results],
        "successful": [r["name"] for r in results if r["response"]], "failed": [r["name"] for r in results if r["error"]],
        "times": {r["name"]: r["time"] for r in results}, "responses": {r["name"]: r["response"] for r in results if r["response"]},
        "synthesis": synthesis if not is_arena else None,
    }
    if not st.session_state.query_history or st.session_state.query_history[-1]["full_prompt"] != user_prompt:
        st.session_state.query_history.append(history_entry)

    with st.expander("📊 Response Time Comparison"):
        max_time = max(r["time"] for r in results) if results else 1
        for r in results:
            cfg = MODELS[r["name"]]
            bar_pct = r["time"] / max_time if max_time > 0 else 0
            label = "✅" if not r["error"] else "❌"
            st.markdown(f"**{cfg['icon']} {r['name']}** — {r['time']}s  ·  {r['words']} words {label}")
            st.progress(min(bar_pct, 1.0))

elif (analyze_clicked or broadcast_clicked) and not user_prompt.strip():
    st.warning("Please enter a prompt first.")

# ──────────────────────────────────────────────
# 7 · QUERY HISTORY  (moved to sidebar — see 💬 History section)
# ──────────────────────────────────────────────
