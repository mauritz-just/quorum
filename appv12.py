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
import uuid as _uuid
from datetime import datetime
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

from db import (init_db, save_history_entry, load_history, clear_history,
                save_meta_prompt, load_meta_prompt,
                save_project, load_projects, save_project_entry,  # kept for DB compat (not used in UI)
                load_project_entries, delete_project)
from auth import require_auth, logout
from key_manager import (
    get_keys, build_models_dict, save_key, delete_key,
    toggle_key, test_key, PROVIDER_PRESETS,
    MAX_ACTIVE_KEYS, MAX_STORED_KEYS, MAX_KEYS_PER_USER,
    count_active_keys, mask_key, build_key_display_name,
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

# OpenAI o5-mini — key from Streamlit secrets (preferred) or .env fallback
_openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
FALLBACK_MODELS["OpenAI · o5-mini"] = {
    "api_key": _openai_key,
    "endpoint": "https://api.openai.com/v1/chat/completions",
    "model_id": "o5-mini",
    "type": "openai_compat",
    "icon": "🟢",
    "color": "#10A37F",
    "provider": "OpenAI",
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

def _smart_truncate(text: str, max_len: int = 55) -> str:
    """Truncate at word boundary, appending ellipsis."""
    if len(text) <= max_len:
        return text
    truncated = text[:max_len - 1].rsplit(" ", 1)[0]
    return truncated + "…"


class PromptAnalyzer:
    """
    Object-oriented prompt diagnostic engine.

    Encapsulates the seven-dimension scoring framework (Explicit Direction,
    Defined Output, Context & Motivation, Specificity, Uncertainty Safeguard,
    Chain of Thought, Positive Framing) together with the lexicons and
    regex patterns each scorer relies on.

    Each `_score_*` method returns a dict {score, evidence, tip}; the
    public `analyse()` method aggregates them with category-weighted
    scoring into the full analysis report consumed by the UI.
    """

    # ── Lexicons & patterns as class-level constants ──
    ACTION_VERBS = {
        "write", "analyze", "analyse", "generate", "create", "design", "build",
        "summarize", "summarise", "extract", "compare", "evaluate", "draft",
        "translate", "classify", "rewrite", "refactor", "debug", "explain",
        "describe", "list", "calculate", "compute", "review", "critique",
        "identify", "recommend", "plan", "outline", "propose",
    }

    PREAMBLE_PATTERNS = [
        r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood (morning|afternoon|evening)\b",
        r"\bcould you (please )?", r"\bcan you (please )?", r"\bwould you (please )?",
        r"\bplease\b.*\bhelp", r"\bi was wondering\b", r"\bif you don'?t mind\b",
        r"\bi'?d like (you )?to\b", r"\bi want (you )?to\b",
        r"\bthank(s| you)\b", r"\bsorry (to bother|for)",
    ]

    OUTPUT_FORMAT_HINTS = [
        "json", "yaml", "xml", "csv", "markdown", "table", "bullet", "numbered list",
        "paragraph", "prose", "essay", "report", "summary", "outline", "code block",
        "in tags", "<thinking>", "step by step", "step-by-step",
        "word count", "words long", "sentences", "pages", "paragraphs",
        "format:", "structure:", "return only", "respond with only", "output:",
    ]

    CONTEXT_MARKERS = [
        "because", "so that", "in order to", "the goal is", "the purpose is",
        "the audience is", "for a", "for an", "intended for", "will be used for",
        "context:", "background:", "i'm a", "i am a", "we're", "we are",
        "as part of", "for my", "for our", "to help",
    ]

    UNCERTAINTY_SAFEGUARDS = [
        "if you don't know", "if you are unsure", "if uncertain",
        "say so", "admit", "acknowledge", "don't speculate", "do not speculate",
        "don't guess", "do not guess", "don't make up", "rather than guessing",
        "null if unsure", "say 'i don't know'", "express uncertainty",
        "flag any assumptions",
    ]

    COT_TRIGGERS = [
        "think step by step", "step-by-step", "step by step",
        "first", "then", "finally",
        "<thinking>", "reason through", "walk through your reasoning",
        "show your work", "explain your reasoning",
    ]

    NEGATIVE_FRAMING = [
        r"\bdon'?t\b", r"\bdo not\b", r"\bnever\b", r"\bno (bullet|markdown|emoji|preamble)",
        r"\bavoid\b", r"\bexclude\b", r"\bwithout\b",
    ]

    VAGUE_TERMS = {
        "thing", "stuff", "something", "anything", "good", "nice", "better",
        "some", "a lot", "etc", "and so on", "various", "relevant",
    }

    DIMENSION_WEIGHTS = {
        "Explicit Direction": 0.18, "Defined Output": 0.18, "Context & Motivation": 0.16,
        "Specificity": 0.20, "Uncertainty Safeguard": 0.10, "Chain of Thought": 0.10,
        "Positive Framing": 0.08,
    }

    def __init__(self, text):
        self.text = text
        self.lower = text.lower()
        self.tokens = re.findall(r"[a-zA-Z']+", self.lower)
        self.word_count = len(text.split())
        self.sentence_count = max(1, len(re.split(r'[.!?]+', text.strip())) - 1 or 1)

    # ── Seven dimension scorers ──
    def _score_direction(self):
        first_words = self.tokens[:4]
        has_action_verb_early = any(w in self.ACTION_VERBS for w in first_words)
        preamble_hits = [p for p in self.PREAMBLE_PATTERNS if re.search(p, self.lower)]
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

    def _score_output_definition(self):
        """
        Layer-1 fix: distinguish *mentioning* a format from *specifying a schema*.
        Bare 'give me JSON' no longer earns the full score — the prompt must also
        supply field names, an example, code-block delimiters, or a quoted schema.
        """
        hits = [h for h in self.OUTPUT_FORMAT_HINTS if h in self.lower]
        has_format_word = any(h in self.lower for h in ["json", "yaml", "xml", "table", "bullet", "markdown", "csv"])
        has_length = bool(re.search(r"\b(\d+)\s*(words?|sentences?|paragraphs?|bullets?|items?|lines?|pages?)\b", self.lower))
        has_structure_word = any(h in self.lower for h in ["format:", "structure:", "return only", "respond with only"])

        # NEW — does the prompt actually show / describe the schema?
        has_code_block = "```" in self.text or "    " in self.text  # fenced or indented block
        has_field_markers = bool(re.search(r"[{\[].*[:,].*[}\]]", self.text))  # JSON-ish dict/array
        # Field-style lines: "fields:", "- name:", "* type:", "title:", etc.
        has_field_list = bool(re.search(r"(?im)^\s*[-*]\s*\w+\s*[:\-]", self.text)) or "fields:" in self.lower
        has_example_keyword = any(k in self.lower for k in ["example:", "for example", "e.g.", "such as:", "like this:"])
        has_real_schema = has_code_block or has_field_markers or has_field_list or has_example_keyword

        score = 0
        if has_format_word and has_real_schema:
            score += 50          # format named AND schema/example provided
        elif has_format_word:
            score += 20          # format mentioned only — partial credit
        if has_length:
            score += 25
        if has_structure_word:
            score += 25
        if hits and score < 25:
            score = 25

        evidence = []
        if has_format_word and has_real_schema:
            evidence.append("Format specified with concrete schema / example")
        elif has_format_word:
            evidence.append(f"Format named ({', '.join(hits[:2])}) but no schema or example given")
        if has_length:
            evidence.append("Explicit length constraint present")
        if not hits and not has_length:
            evidence.append("No format, length, or structure specified")

        tip = None
        if score < 50:
            if has_format_word and not has_real_schema:
                tip = "You named the format but didn't show its shape — add field names, an example, or a code-block schema."
            else:
                tip = "State the exact output shape — format (JSON / bullets / prose), length, and what sections it must contain."
        return {"score": min(score, 100), "evidence": evidence, "tip": tip}

    def _score_context(self):
        has_why = any(m in self.lower for m in ["because", "so that", "in order to", "the goal"])
        has_audience = any(m in self.lower for m in ["audience", "for a", "for an", "intended for"])
        has_use_case = any(m in self.lower for m in ["will be used for", "as part of", "for my", "to help"])
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

    def _score_specificity(self):
        """
        Layer-1 fix: proper nouns only count when they appear as *constraints*
        (after 'using/in/with/for/as/on/via/targeting/written in/built with' etc.)
        rather than incidental topic mentions.
        """
        numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", self.text)
        vague_hits = [v for v in self.VAGUE_TERMS if v in self.tokens]

        # NEW — only credit proper nouns that follow a constraint preposition.
        constraint_pattern = r"\b(using|use|in|with|for|as|on|via|targeting|written in|built with|via|by|through|based on)\s+([A-Z][a-zA-Z0-9_.\-]+(?:\s+[A-Z][a-zA-Z0-9_.\-]+){0,2})"
        constraint_entities = re.findall(constraint_pattern, self.text)
        # Deduplicate the captured entity (group 2)
        entity_names = list({m[1] for m in constraint_entities})

        score = 20
        score += min(len(numbers) * 12, 30)
        score += min(len(entity_names) * 10, 20)   # NEW — constraints, not bare proper nouns
        score -= min(len(vague_hits) * 10, 30)
        if len(self.tokens) >= 25:
            score += 20
        score = max(0, min(score, 100))

        evidence = []
        if numbers:
            evidence.append(f"{len(numbers)} concrete number(s) / quantit(ies)")
        if entity_names:
            evidence.append(f"Constraint entities: {', '.join(entity_names[:3])}")
        if vague_hits:
            evidence.append(f"Vague terms to sharpen: {', '.join(vague_hits[:3])}")
        tip = None
        if vague_hits:
            tip = f"Replace vague words ({', '.join(vague_hits[:2])}) with concrete constraints."
        elif score < 50:
            tip = "Add concrete constraints: exact numbers, specific tools/libraries/frameworks (named after 'using', 'in', 'with')."
        return {"score": score, "evidence": evidence, "tip": tip}

    def _score_uncertainty(self):
        hits = [s for s in self.UNCERTAINTY_SAFEGUARDS if s in self.lower]
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

    def _score_cot(self, needs_reasoning):
        """
        Layer-1 fix: CoT is required when the *task* needs reasoning (math,
        multi-step, comparison, trade-off, derivation), not when the prompt
        merely happens to be long.
        """
        hits = [t for t in self.COT_TRIGGERS if t in self.lower]
        has_cot = len(hits) > 0
        if not needs_reasoning:
            return {"score": 100, "evidence": ["Task doesn't require step-by-step reasoning"], "tip": None}
        if has_cot:
            return {"score": 100, "evidence": [f"CoT trigger present: '{hits[0]}'"], "tip": None}
        return {"score": 30, "evidence": ["Reasoning-heavy task without step-by-step trigger"],
                "tip": "This task needs reasoning — add 'Think step by step before answering' or structure it as First / Then / Finally."}

    def _score_framing(self, has_positive_direction):
        """
        Layer-1 fix: 'don't / no / avoid' constraints are only harmful when the
        prompt has no positive specification alongside them. Prompts that say
        BOTH what to do AND what to avoid are well-formed — the rule of thumb
        is 'replace negatives with positives', not 'never use negatives'.
        """
        negative_hits = sum(1 for p in self.NEGATIVE_FRAMING if re.search(p, self.lower))
        total_sentences = max(1, len(re.split(r"[.!?]+", self.text.strip())))
        ratio = negative_hits / total_sentences

        if negative_hits == 0:
            score = 100
        elif has_positive_direction:
            # Negatives alongside positives are fine — small or no penalty.
            score = 90 if ratio < 0.4 else 70
        else:
            # Negatives without any positive direction is the real anti-pattern.
            if ratio < 0.3:
                score = 60
            elif ratio < 0.6:
                score = 35
            else:
                score = 15

        evidence = []
        if negative_hits == 0:
            evidence.append("Uses positive / architectural framing")
        elif has_positive_direction:
            evidence.append(f"{negative_hits} negative constraint(s), balanced by positive direction")
        else:
            evidence.append(f"{negative_hits} negative constraint(s) ('don't / no / avoid') with no positive specification")

        tip = None
        if score < 60:
            tip = "Replace 'don't do X' with 'do Y instead'. Negative constraints leave the output space underspecified."
        return {"score": score, "evidence": evidence, "tip": tip}

    # ── Complexity & category helpers ──
    def _compute_complexity(self):
        """
        Layer-1 fix: word count is now a *minor* signal (max 15 pts of 100).
        Real complexity comes from task-difficulty signals: reasoning required,
        multi-step structure, technical domain, comparative analysis,
        constraint count, and ambiguity flags.

        Returns (complexity, signals_dict). The signals_dict is consumed by
        the CoT scorer so it can decide whether reasoning is actually needed.
        """
        # ── Task-type signals (mostly your original keyword sets, kept stable) ──
        has_code = any(kw in self.lower for kw in ["code", "function", "script", "program", "debug",
                                                    "python", "javascript", "sql", "html", "css", "api"])
        has_creative = any(kw in self.lower for kw in ["write", "story", "poem", "essay", "blog",
                                                        "creative", "imagine", "describe", "narrative"])
        has_analysis = any(kw in self.lower for kw in ["compare", "analyse", "analyze", "explain",
                                                       "evaluate", "pros and cons", "difference", "versus", "vs"])

        # ── NEW: reasoning signals — the real driver of "needs CoT" ──
        has_math = bool(re.search(r"\b(calculate|compute|solve|derive|prove|estimate|optimi[sz]e)\b", self.lower)) \
                   or bool(re.search(r"\d+\s*[+\-*/×÷=]\s*\d+", self.text))
        has_multistep = bool(re.search(r"\b(step|first.*then|then.*finally|stages?|phases?|workflow|pipeline)\b", self.lower))
        has_tradeoff = any(kw in self.lower for kw in ["trade-off", "tradeoff", "trade off", "pros and cons",
                                                        "advantages and disadvantages", "weigh", "balance between"])
        has_why = bool(re.search(r"\b(why|how come|reason|justify|because)\b", self.lower))
        has_compare = bool(re.search(r"\b(compare|contrast|versus|vs\.?|difference between|better than)\b", self.lower))

        # ── Ambiguity signals (raise complexity — harder to satisfy) ──
        has_conditional = bool(re.search(r"\b(if|unless|when|depending on|given that)\b", self.lower))
        has_multiple_outputs = bool(re.search(r"\b(and also|as well as|in addition|plus|furthermore)\b", self.lower))

        # ── Score build ──
        complexity = 5
        # Word count is now a *minor* tiebreaker, not the main signal.
        complexity += min(self.word_count * 0.4, 15)

        # Task type weight
        complexity += 18 if has_code else 0
        complexity += 8 if has_creative else 0

        # Reasoning weight — these are the heavy hitters now.
        complexity += 18 if has_math else 0
        complexity += 14 if has_compare else 0
        complexity += 12 if has_tradeoff else 0
        complexity += 12 if has_multistep else 0
        complexity += 10 if has_analysis else 0
        complexity += 8 if has_why else 0

        # Ambiguity adds to difficulty.
        complexity += 8 if has_conditional else 0
        complexity += 6 if has_multiple_outputs else 0
        complexity += 5 if self.sentence_count > 4 else 0

        complexity = min(int(complexity), 100)

        signals = {
            "has_code": has_code,
            "has_creative": has_creative,
            "has_analysis": has_analysis,
            "has_math": has_math,
            "has_multistep": has_multistep,
            "has_tradeoff": has_tradeoff,
            "has_why": has_why,
            "has_compare": has_compare,
            # A task "needs reasoning" if any of these explicit signals are present.
            "needs_reasoning": has_math or has_multistep or has_tradeoff or has_compare or (has_analysis and has_why),
        }
        return complexity, signals

    def _categorise(self, has_code, has_creative, has_analysis):
        has_question = "?" in self.text
        if has_code:
            return "💻 Code / Technical"
        if has_creative:
            return "✍️ Creative / Writing"
        if has_analysis:
            return "🔍 Analysis / Comparison"
        if has_question:
            return "❓ Question / Factual"
        return "💬 General"

    # ── Public entry point ──
    def analyse(self):
        complexity, signals = self._compute_complexity()
        if complexity < 30:
            level = "Simple"
        elif complexity < 60:
            level = "Moderate"
        else:
            level = "Complex"

        # Run direction & output scorers first — their results feed downstream scorers.
        direction = self._score_direction()
        output = self._score_output_definition()
        context = self._score_context()
        specificity = self._score_specificity()
        # "Positive direction" exists if the prompt scored decently on at least one
        # positive-specification dimension (direction, output, or specificity).
        has_positive_direction = direction["score"] >= 50 or output["score"] >= 50 or specificity["score"] >= 50

        dims = {
            "Explicit Direction": direction,
            "Defined Output": output,
            "Context & Motivation": context,
            "Specificity": specificity,
            "Uncertainty Safeguard": self._score_uncertainty(),
            "Chain of Thought": self._score_cot(signals["needs_reasoning"]),
            "Positive Framing": self._score_framing(has_positive_direction),
        }
        quality = int(sum(dims[k]["score"] * w for k, w in self.DIMENSION_WEIGHTS.items()))
        category = self._categorise(signals["has_code"], signals["has_creative"], signals["has_analysis"])
        ranked = sorted([(name, d) for name, d in dims.items() if d["tip"]], key=lambda kv: kv[1]["score"])
        suggestions = [f"**{name}** — {d['tip']}" for name, d in ranked[:4]]
        return {"word_count": self.word_count, "sentence_count": self.sentence_count,
                "complexity": complexity, "complexity_level": level, "quality": quality,
                "category": category, "suggestions": suggestions, "dimensions": dims}


# Backwards-compatible module-level wrapper (used everywhere in the UI layer).
def analyse_prompt(text):
    return PromptAnalyzer(text).analyse()


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


def _auto_refine_mode(analysis):
    """Pick simplify vs enhance based on diagnostic scores."""
    word_count = analysis["word_count"]
    dims = analysis["dimensions"]
    specificity = dims["Specificity"]["score"]
    context = dims["Context & Motivation"]["score"]
    output_def = dims["Defined Output"]["score"]
    detail_avg = (specificity + context + output_def) / 3
    # Long prompt that's already reasonably specific → trim it
    if word_count > 70 and detail_avg >= 50:
        return "simplify"
    return "enhance"


def refine_prompt(user_prompt, analysis, timeout=60, mode="balanced", preferred_analyzer=None):
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
    if preferred_analyzer and preferred_analyzer in MODELS:
        candidates = [preferred_analyzer] + [c for c in candidates if c != preferred_analyzer]
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


def summarise_prompt(text, max_chars=65):
    """Return a compact, meaningful title for a prompt (max max_chars characters)."""
    clean = re.sub(r"[\"'`\n\r]+", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    # Strip common filler openings so the title starts with the real request
    clean = re.sub(
        r"^(please |can you (please )?|could you (please )?|i want you to |"
        r"help me |i'?d like (you )?to |i would like (you )?to |"
        r"i need (you )?to |would you (please )?|i was wondering if you could )",
        "", clean, flags=re.IGNORECASE,
    ).strip()
    if not clean:
        clean = text.strip()
    clean = clean[0].upper() + clean[1:] if clean else clean
    if len(clean) <= max_chars:
        return clean
    truncated = clean[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return truncated + "…"


def build_effective_prompt(user_prompt):
    """Prepend the meta-prompt (if set in Settings) to user_prompt before broadcasting."""
    meta = st.session_state.get("s_meta_prompt", "").strip()
    if not meta:
        return user_prompt
    return f"{meta}\n\n---\n\n{user_prompt}"


def _extract_recommended_answer(text):
    """
    Pull the Recommended Answer block out of synthesis markdown.
    Works line-by-line so it handles all common LLM heading formats:
      ## 1. Recommended Answer
      **1. Recommended Answer**
      1. **Recommended Answer**
      **Recommended Answer**: inline text...
    Returns (answer_text, rest_text). If not found, (None, text).
    """
    lines = text.split("\n")
    header_idx = None
    inline_content = None

    # Patterns that identify a section header line
    _section_re = re.compile(
        r"^(#{1,4}\s+|[*_]{1,2}|\d+\.\s+|[*_]{1,2}\d+\.\s+)*"
        r"(?:\d+\.\s*)?[*_]*Recommended Answer[*_]*\s*:?\s*(.*)",
        re.IGNORECASE,
    )
    # Pattern to detect the START of any other section header line
    _next_section_re = re.compile(
        r"^(#{1,4}\s+|\*{1,2}\d+\.|\d+\.\s+\*{0,2}[A-Z]|\*{2}[A-Z])[^\n]*$",
        re.IGNORECASE,
    )

    for i, line in enumerate(lines):
        m = _section_re.match(line.strip())
        if m:
            header_idx = i
            rest_of_header = m.group(2).strip() if m.group(2) else ""
            inline_content = rest_of_header if rest_of_header else None
            break

    if header_idx is None:
        return None, text

    # Collect answer body lines (everything after the header until next section)
    body_start = header_idx + 1
    end_idx = len(lines)
    for j in range(body_start, len(lines)):
        stripped = lines[j].strip()
        if stripped and _next_section_re.match(stripped):
            end_idx = j
            break

    body_lines = ([inline_content] if inline_content else []) + lines[body_start:end_idx]

    # Strip leading/trailing blank lines from body
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    if not body_lines:
        return None, text

    answer = "\n".join(body_lines).strip()
    before = "\n".join(lines[:header_idx]).strip()
    after = "\n".join(lines[end_idx:]).strip()
    rest = "\n\n".join(p for p in [before, after] if p)
    return answer, rest


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
#
# Object-oriented design:
#
#   • ModelClient (abstract base) defines the contract every provider must honour:
#       generate(prompt, max_tokens, temperature, timeout) -> str
#     plus the public call(...) wrapper which times the request and returns a
#     QueryResult dataclass.
#
#   • Three concrete subclasses (GeminiClient, OpenAICompatClient, AnthropicClient)
#     each implement generate() with the provider-specific payload format and
#     response shape. This is the polymorphism replacing the previous
#     if/elif branching on config["type"].
#
#   • The QueryResult dataclass replaces the implicit-schema result dict;
#     to_dict() is used at the boundary so all downstream code (UI, DB,
#     session history) keeps consuming dicts as before.

@dataclass
class QueryResult:
    """Typed container for one model's response in a broadcast."""
    name: str
    response: str | None
    time: float
    words: int
    error: str | None

    def to_dict(self):
        return asdict(self)


class ModelClient:
    """
    Abstract base for any LLM provider this app can call.

    Subclasses override generate() with provider-specific request/response
    handling. The public call() method wraps generate() with timing and
    exception capture, returning a QueryResult.
    """

    def __init__(self, name, config):
        self.name = name
        self.config = config

    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=90):
        raise NotImplementedError("Subclasses must implement generate()")

    def call(self, prompt, max_tokens=1024, temperature=0.7, timeout=90):
        start = time.time()
        try:
            text = self.generate(prompt, max_tokens, temperature, timeout)
            return QueryResult(name=self.name, response=text,
                               time=round(time.time() - start, 2),
                               words=len(text.split()), error=None)
        except Exception as e:
            return QueryResult(name=self.name, response=None,
                               time=round(time.time() - start, 2),
                               words=0, error=str(e))

    @classmethod
    def from_config(cls, name, config):
        """Factory: dispatch to the right subclass based on config['type']."""
        t = config.get("type", "openai_compat")
        if t == "gemini":
            return GeminiClient(name, config)
        if t == "anthropic":
            return AnthropicClient(name, config)
        return OpenAICompatClient(name, config)


class GeminiClient(ModelClient):
    """Google AI Studio (generateContent) — native Gemini payload shape."""

    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=90):
        cfg = self.config
        url = f"{cfg['endpoint']}?key={cfg['api_key']}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
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


class OpenAICompatClient(ModelClient):
    """Any provider speaking the OpenAI Chat Completions shape (Groq, Mistral, Cerebras, OpenRouter, …)."""

    # OpenAI o-series reasoning models require max_completion_tokens and reject temperature
    _O_SERIES_RE = re.compile(r"^o\d")

    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=90):
        cfg = self.config
        headers = {"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"}
        model_id = cfg.get("model_id", "")
        is_o_series = bool(self._O_SERIES_RE.match(model_id))
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_o_series:
            # Reasoning models: use max_completion_tokens, no temperature
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
            payload["temperature"] = temperature
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class AnthropicClient(ModelClient):
    """Anthropic Messages API — distinct header & response schema."""

    def generate(self, prompt, max_tokens=1024, temperature=0.7, timeout=90):
        cfg = self.config
        headers = {
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": cfg["model_id"],
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(cfg["endpoint"], headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


# ── Backwards-compatible function wrappers ──
# These keep every existing call site working unchanged: refiner, aggregator,
# UI, history — they all still call _call_gemini / _call_openai_compat / etc.
# Internally they delegate to the new ModelClient subclasses.

def _call_gemini(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    return GeminiClient("Gemini", config).generate(prompt, max_tokens, temperature, timeout)


def _call_openai_compat(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    return OpenAICompatClient("OpenAICompat", config).generate(prompt, max_tokens, temperature, timeout)


def _call_anthropic(prompt, config, max_tokens=1024, temperature=0.7, timeout=90):
    return AnthropicClient("Anthropic", config).generate(prompt, max_tokens, temperature, timeout)


def call_model(name, prompt, max_tokens=1024, temperature=0.7, timeout=90):
    """Top-level convenience: build the right client from MODELS[name] and call it."""
    config = MODELS[name]
    client = ModelClient.from_config(name, config)
    return client.call(prompt, max_tokens, temperature, timeout).to_dict()


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

YOUR TASK — produce a structured synthesis in this exact order:

1. **Recommended Answer**: Write the single best answer you would give the user, drawing on all models. Make it complete and immediately useful — this is the primary output.

2. **Consensus Summary**: What do most or all models agree on? Write a clear, merged answer that captures the shared knowledge. Remove redundancies.

3. **Unique Insights**: List any valuable points that only one or two models mentioned and that add value.

4. **Disagreements**: Note any areas where models contradicted each other. State each position fairly.

5. **Confidence Score**: Rate from 1-10 how confident the combined answer is (10 = perfect agreement on a factual matter, 1 = wild disagreement).

Respond in clean Markdown."""


def aggregate_responses(user_prompt, results, preferred_aggregator=None):
    valid = [r for r in results if r["response"]]
    if len(valid) < 2:
        return None

    responses_block = ""
    for r in valid:
        responses_block += f"\n--- {r['name']} ---\n{r['response']}\n"

    meta_prompt = AGGREGATION_PROMPT.format(count=len(valid), user_prompt=user_prompt, responses_block=responses_block)

    # ── Attempt 0: User-selected aggregator model ──
    if preferred_aggregator and preferred_aggregator in MODELS:
        _agg_cfg = MODELS[preferred_aggregator]
        try:
            if _agg_cfg["type"] == "gemini":
                _agg_text = _call_gemini(meta_prompt, _agg_cfg, max_tokens=2048, temperature=0.3, timeout=120)
            elif _agg_cfg["type"] == "anthropic":
                _agg_text = _call_anthropic(meta_prompt, _agg_cfg, max_tokens=2048, temperature=0.3, timeout=120)
            else:
                _agg_text = _call_openai_compat(meta_prompt, _agg_cfg, max_tokens=2048, temperature=0.3, timeout=120)
            return _agg_text + f"\n\n---\n*Aggregated by {preferred_aggregator}*"
        except Exception:
            pass  # fall through to built-in fallback chain

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

# ── Load persisted user data once per session (survives refresh & re-login) ──
if "s_meta_prompt" not in st.session_state:
    st.session_state.s_meta_prompt = load_meta_prompt(_user_id)
if "query_history" not in st.session_state:
    st.session_state.query_history = load_history(_user_id)

# ── Projects / Sessions ──
if "projects" not in st.session_state:
    _db_projects = load_projects(_user_id)
    if _db_projects:
        st.session_state.projects = []
        for _p in _db_projects:
            _p["entries"] = load_project_entries(_user_id, _p["id"])
            st.session_state.projects.append(_p)
    else:
        # First run: create a default project
        _default_id = str(_uuid.uuid4())
        _default_proj = {
            "id": _default_id,
            "title": "New Project",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "entries": [],
        }
        st.session_state.projects = [_default_proj]
        save_project(_user_id, _default_id, "New Project")

if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = st.session_state.projects[0]["id"]

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
    /* Compact Prompt Log sidebar buttons */
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        font-size: 0.78rem !important;
        line-height: 1.3 !important;
        min-height: unset !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Value Proposition + Architects (popover — above the header bar) ──
_VP_TEXT = (
    "Our tool earns a spot in your daily workflow for one of three reasons. "
    "**Cross-verification** for students and researchers. "
    "**Best answer, guaranteed** for professionals. "
    "**Cost efficiency** for companies with multiple AI subscriptions. "
    "We run all models **in parallel**, add a **synthesis layer** with confidence scores, "
    "make **disagreement transparent**, and do it all in **one prompt box**."
)
_VP_ARCHITECTS = "1. Jan Schützenmeister\n2. Mauritz Just\n3. Leandra Sühling\n4. Malin Minten"
_, _vp_btn_col = st.columns([5, 1])
with _vp_btn_col:
    if hasattr(st, "popover"):
        with st.popover("💡 About", use_container_width=True):
            st.markdown(_VP_TEXT)
            st.markdown("---")
            st.markdown("**Architects**")
            st.markdown(_VP_ARCHITECTS)
    else:
        with st.expander("💡 About"):
            st.markdown(_VP_TEXT)
            st.markdown("---")
            st.markdown("**Architects**")
            st.markdown(_VP_ARCHITECTS)

st.markdown('<div class="top-bar"><h1>🏟️ QuorumAI — Model Arena & Aggregator</h1><span>Multi-LLM Aggregation Platform · Compare · Synthesise · Decide</span></div>', unsafe_allow_html=True)

# ── Settings / generation defaults (must be above sidebar) ──
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

for key, default in [("s_max_tokens", 1024), ("s_temperature", 0.7), ("s_timeout", 240), ("s_show_analysis", True)]:
    if key not in st.session_state:
        st.session_state[key] = default

def _save(src, dst):
    st.session_state[dst] = st.session_state[src]

def _save_meta():
    st.session_state.s_meta_prompt = st.session_state["_w_meta_prompt"]
    save_meta_prompt(_user_id, st.session_state.s_meta_prompt)

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
    if st.button("✕ Close Settings" if st.session_state.show_settings else "⚙️ Settings",
                 use_container_width=True, key="settings_toggle"):
        st.session_state.show_settings = not st.session_state.show_settings
        st.rerun()

    if st.session_state.show_settings:
        # ── SETTINGS PANEL ──
        mode = st.session_state.get("_last_mode", "🏟️ Arena (side-by-side)")
        selected_models = st.session_state.get("_last_selected_models", [])

        st.markdown("---")
        st.markdown("### 🛠️ Generation Settings")
        st.slider("Max response length (tokens)", 100, 5000, st.session_state.s_max_tokens, 100,
                  key="_w_mt", on_change=lambda: _save("_w_mt", "s_max_tokens"))
        st.slider("Temperature (creativity)", 0.0, 1.5, st.session_state.s_temperature, 0.1,
                  key="_w_temp", on_change=lambda: _save("_w_temp", "s_temperature"))
        st.slider("Timeout per model (seconds)", 30, 600, st.session_state.s_timeout, 30,
                  key="_w_to", on_change=lambda: _save("_w_to", "s_timeout"))
        st.toggle("Show Prompt Analysis", value=st.session_state.s_show_analysis,
                  key="_w_sa", on_change=lambda: _save("_w_sa", "s_show_analysis"))
        st.text_area(
            "Meta-prompt (prepended to every broadcast)",
            value=st.session_state.s_meta_prompt,
            key="_w_meta_prompt",
            height=90,
            placeholder="e.g. Always respond in German. Be concise. Use bullet points.",
            on_change=_save_meta,
        )

        st.markdown("---")
        st.markdown("### 🔑 My API Keys")
        _user_keys = st.session_state.user_keys
        _stored_count = len(_user_keys)
        _active_key_count = sum(1 for k in _user_keys if k["is_active"])
        st.caption(f"{_stored_count} key{'s' if _stored_count != 1 else ''} stored · {_active_key_count} active")

        for _k in _user_keys:
            _preset_k = PROVIDER_PRESETS.get(_k["provider_name"], {})
            _icon_k = _preset_k.get("icon", "🔑")
            _kc1, _kc2, _kc3 = st.columns([4, 1, 1])
            with _kc1:
                _status_k = "✅" if _k["is_active"] else "⏸"
                _pretty_k = build_key_display_name(_k["display_name"], _k["provider_name"], _k["model_id"])
                st.markdown(f"{_icon_k} **{_pretty_k}** {_status_k}")
                st.caption(f"`{_k['masked_key']}`")
            with _kc2:
                _tog_lbl = "Pause" if _k["is_active"] else "Enable"
                if st.button("⏸" if _k["is_active"] else "▶", key=f"tog_{_k['id']}",
                             help=_tog_lbl):
                    try:
                        toggle_key(_user_id, _k["id"])
                        st.session_state.user_keys = get_keys(_user_id)
                        st.rerun()
                    except ValueError as _e:
                        st.error(str(_e))
            with _kc3:
                if st.button("🗑", key=f"del_{_k['id']}", help="Delete key"):
                    delete_key(_user_id, _k["id"])
                    st.session_state.user_keys = get_keys(_user_id)
                    st.rerun()

        if "add_key_form_gen" not in st.session_state:
            st.session_state.add_key_form_gen = 0
        _gen = st.session_state.add_key_form_gen

        with st.expander("➕ Add API Key"):
            _provider = st.selectbox("Provider", list(PROVIDER_PRESETS.keys()), key=f"new_key_provider_{_gen}")
            _preset_cfg = PROVIDER_PRESETS[_provider]
            if _preset_cfg["models"]:
                _model_id = st.selectbox("Model", _preset_cfg["models"], key=f"new_key_model_{_gen}")
            else:
                _model_id = st.text_input("Model ID", key=f"new_key_model_custom_{_gen}")
            _endpoint = _preset_cfg["endpoint"]
            if not _endpoint:
                _endpoint = st.text_input("Endpoint URL", key=f"new_key_endpoint_{_gen}")
            _display_name = st.text_input("Display name (optional)", key=f"new_key_display_{_gen}")

            _show_key = st.toggle("👁 Show key", value=False, key=f"show_new_key_{_gen}")
            if not _show_key:
                st.markdown("""
                <style>
                  div[data-testid="stTextInput"] input[aria-label="API Key"] {
                    -webkit-text-security: disc !important;
                    -moz-text-security: disc !important;
                    text-security: disc !important;
                  }
                </style>
                """, unsafe_allow_html=True)
            _api_key_input = st.text_input(
                "API Key",
                type="default",
                key=f"new_key_value_{_gen}",
                placeholder="Paste your API key here",
            )

            if _display_name.strip():
                _norm = _display_name.strip().lower()
                _dupe_name = any(
                    (k["display_name"] or k["provider_name"]).strip().lower() == _norm
                    for k in _user_keys
                )
                if _dupe_name:
                    st.warning("An API with this name already exists.")

            if st.button("🧪 Test & Save", key=f"save_new_key_{_gen}", use_container_width=True):
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
                            st.session_state.add_key_form_gen += 1
                            st.success("✅ Key saved!")
                            st.rerun()
                        except ValueError as _e:
                            st.error(str(_e))
                    else:
                        st.error(f"❌ Key test failed: {_msg}")

    else:
        # ── NORMAL SIDEBAR CONTENT ──
        st.markdown("---")
        mode = st.radio("Mode", ["🏟️ Arena (side-by-side)", "🧠 Aggregator (synthesised)"],
                        index=0, key="mode_radio")
        st.session_state._last_mode = mode
        st.markdown("---")

        # ── Model Roles ──
        st.markdown("### 🎛️ Model Roles")
        _role_options = ["(auto)"] + list(MODELS.keys())

        def _default_role_model():
            for _rn in MODELS:
                if "o5-mini" in _rn or MODELS[_rn].get("model_id", "") == "o5-mini":
                    return _rn
            return "(auto)"

        if "s_analyzer_model" not in st.session_state:
            st.session_state.s_analyzer_model = "(auto)"
        if "s_aggregator_model" not in st.session_state:
            st.session_state.s_aggregator_model = _default_role_model()  # defaults to o5-mini if available

        _ana_idx = _role_options.index(st.session_state.s_analyzer_model) if st.session_state.s_analyzer_model in _role_options else 0
        _agg_idx = _role_options.index(st.session_state.s_aggregator_model) if st.session_state.s_aggregator_model in _role_options else 0
        st.session_state.s_analyzer_model = st.selectbox(
            "Prompt Analyzer Model", _role_options, index=_ana_idx, key="ana_model_sel",
            help="Model used to rewrite prompts when you click Refine"
        )
        st.session_state.s_aggregator_model = st.selectbox(
            "Aggregator Model", _role_options, index=_agg_idx, key="agg_model_sel",
            help="Model used to synthesise responses in Aggregator mode"
        )

        # ── Output Models ──
        st.markdown("### 🤖 Output Models")
        st.caption(f"Select up to {MAX_ACTIVE_KEYS} models to receive your prompt:")
        _all_model_names = list(MODELS.keys())
        _standard_names = [n for n in _all_model_names if n in FALLBACK_MODELS]
        _added_names = [n for n in _all_model_names if n not in FALLBACK_MODELS]

        for _idx, _name in enumerate(_all_model_names):
            _ckey = f"chk_{_name}"
            if _ckey not in st.session_state:
                st.session_state[_ckey] = (_idx < MAX_ACTIVE_KEYS)

        _checked_now = [n for n in _all_model_names if st.session_state.get(f"chk_{n}", False)]
        if len(_checked_now) > MAX_ACTIVE_KEYS:
            for _xname in _checked_now[MAX_ACTIVE_KEYS:]:
                st.session_state[f"chk_{_xname}"] = False

        _live_active_count = sum(1 for n in _all_model_names if st.session_state.get(f"chk_{n}", False))
        _at_max = _live_active_count >= MAX_ACTIVE_KEYS

        selected_models = []

        if _standard_names:
            st.caption("**Standard Models**")
            for _name in _standard_names:
                _cfg = MODELS[_name]
                _is_active = st.session_state.get(f"chk_{_name}", False)
                _disabled = _at_max and not _is_active
                _help = f"Max {MAX_ACTIVE_KEYS} at once — deselect one first" if _disabled else None
                if st.checkbox(f"{_cfg['icon']} {_name}", key=f"chk_{_name}", disabled=_disabled, help=_help):
                    selected_models.append(_name)

        if _added_names:
            st.caption("**Added Models**")
            for _name in _added_names:
                _cfg = MODELS[_name]
                _is_active = st.session_state.get(f"chk_{_name}", False)
                _disabled = _at_max and not _is_active
                _help = f"Max {MAX_ACTIVE_KEYS} at once — deselect one first" if _disabled else None
                if st.checkbox(f"{_cfg['icon']} {_name}", key=f"chk_{_name}", disabled=_disabled, help=_help):
                    selected_models.append(_name)

        if not selected_models:
            st.error("Select at least 1 model.")
        else:
            st.caption(f"{len(selected_models)} of {len(_all_model_names)} models selected")

        st.session_state._last_selected_models = selected_models

        # ── Prompt Log ──
        st.markdown("---")
        st.markdown("### 📜 Prompt Log")

        _log = st.session_state.get("query_history") or []
        if not _log:
            st.caption("No runs yet — submit a prompt to get started.")
        else:
            for _lidx, _entry in enumerate(reversed(_log)):
                _real_idx = len(_log) - 1 - _lidx
                _lmode = _entry.get("mode", "")
                _lmode_icon = "🏟️" if _lmode == "Arena" else ("🧠" if _lmode == "Aggregator" else "✨")
                _raw_title = (_entry.get("summary") or _entry.get("prompt") or "Query")
                _ltitle = _smart_truncate(_raw_title, 55)
                _lts = _entry.get("timestamp", "")
                _llabel = f"{_lmode_icon} {_ltitle} · {_lts}"

                if st.button(_llabel, key=f"plog_{_real_idx}", use_container_width=True):
                    st.session_state._prompt_to_apply = _entry["full_prompt"]
                    st.session_state.refinement_history = []
                    st.session_state.last_analysis = _entry.get("analysis")

                    _resp  = _entry.get("responses") or {}
                    _errs  = _entry.get("errors") or {}
                    _times = _entry.get("times") or {}
                    _wcs   = _entry.get("word_counts") or {}
                    _all_models = _entry.get("models_used") or list(_resp.keys())
                    _has_timing = bool(_times and any(v and v > 0 for v in _times.values()))

                    _entry_results = []
                    for _mname in _all_models:
                        _entry_results.append({
                            "name":     _mname,
                            "response": _resp.get(_mname),
                            "error":    _errs.get(_mname),
                            "time":     _times.get(_mname, 0),
                            "words":    _wcs.get(_mname) or len((_resp.get(_mname) or "").split()),
                        })

                    _syn = _entry.get("synthesis")
                    st.session_state.broadcast_output = {
                        "synthesis":    _syn,
                        "results":      _entry_results,
                        "mode":         _entry.get("mode", "Aggregator"),
                        "prompt":       _entry["full_prompt"],
                        "quality":      _entry.get("quality", 0),
                        "complexity":   _entry.get("complexity", ""),
                        "n_models":     len(_entry_results),
                        "has_timing":   _has_timing,
                        "from_history": True,
                        "timestamp":    _entry.get("timestamp", ""),
                        "summary":      _entry.get("summary", ""),
                    } if (_entry_results or _syn) else None
                    st.rerun()

        st.markdown("")
        if _log:
            if st.button("🗑️ Clear Prompt Log", use_container_width=True, key="clear_prompt_log"):
                clear_history(_user_id)
                st.session_state.query_history = []
                st.session_state.broadcast_output = None
                st.rerun()

# ── Resolve settings vars from session_state (used throughout main area) ──
max_tokens = st.session_state.s_max_tokens
temperature = st.session_state.s_temperature
timeout_sec = st.session_state.s_timeout
show_prompt_analysis = st.session_state.s_show_analysis

# ── Main Area ──
for key, default in [
    ("working_prompt", ""),
    ("last_analysis", None),
    ("refinement_history", []),
    ("broadcast_now", False),
    ("_pending_refined", None),
    ("_pending_analysis", None),
    ("_prompt_to_apply", None),
    ("_loaded_history_entry", None),
    ("broadcast_output", None),
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

# ── Live model recommendation (auto-computed, always visible when prompt exists) ──
if user_prompt.strip() and selected_models:
    _live_analysis = analyse_prompt(user_prompt)
    _rec = recommend_models(_live_analysis, selected_models)

    _suggested_pills = "  ".join(
        f'<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:0.82rem;font-weight:600">'
        f'{MODELS[m]["icon"]} {m}</span>'
        for m in _rec["suggested"]
    )
    _skipped = [m for m in selected_models if m not in _rec["suggested"]]
    _skipped_pills = "  ".join(
        f'<span style="background:#1e1e2e;color:#475569;padding:3px 10px;border-radius:12px;font-size:0.82rem">'
        f'{MODELS[m]["icon"]} {m}</span>'
        for m in _skipped
    ) if _skipped else ""

    _complexity_color = {"Simple": "#22c55e", "Moderate": "#f59e0b", "Complex": "#ef4444"}
    _cc = _complexity_color.get(_live_analysis["complexity_level"], "#94a3b8")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #334155;
                border-left:4px solid {_cc};border-radius:10px;padding:0.9rem 1.2rem;margin:0.6rem 0;">
        <div style="display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;">
            <span style="font-size:1.2rem">{_rec["icon"]}</span>
            <span style="font-weight:700;color:#e2e8f0;font-size:0.95rem">{_rec["label"]}</span>
            <span style="color:#64748b;font-size:0.8rem">·</span>
            <span style="color:#94a3b8;font-size:0.82rem">{_live_analysis["complexity_level"]} prompt
              ({_live_analysis["complexity"]}/100)</span>
            <span style="color:#64748b;font-size:0.8rem">·</span>
            <span style="color:#94a3b8;font-size:0.82rem">Quality {_live_analysis["quality"]}/100</span>
        </div>
        <div style="margin-top:0.5rem;font-size:0.8rem;color:#64748b">{_rec["reasoning"]}</div>
        <div style="margin-top:0.6rem;display:flex;gap:0.4rem;flex-wrap:wrap;align-items:center">
            <span style="color:#64748b;font-size:0.78rem;margin-right:4px">Recommended:</span>
            {_suggested_pills}
            {"<span style='color:#475569;font-size:0.78rem;margin-left:8px'>skip: " + _skipped_pills + "</span>" if _skipped_pills else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])
with col_a:
    analyze_clicked = st.button("🔎 Analyze prompt", use_container_width=True, disabled=not user_prompt.strip())
with col_b:
    broadcast_clicked = st.button("📡 Broadcast to selected Models", type="primary", use_container_width=True, disabled=not user_prompt.strip() or not selected_models)

if not selected_models:
    st.warning("Select at least one model in the sidebar.")


# ── Helper: render diagnostic + refinement section ──
def _render_diagnostic_section(analysis, user_prompt):
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
    for i, (dname, d) in enumerate(dim_items):
        col = left if i % 2 == 0 else right
        with col:
            score = d["score"]
            emoji = "🟢" if score >= 75 else ("🟡" if score >= 45 else "🔴")
            with st.expander(f"{emoji} **{dname}** — {score}/100", expanded=False):
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

    # ── Refinement ──
    st.markdown("---")
    st.markdown("### ✨ Refine your prompt")
    st.caption("AI picks the right strategy automatically — simplify if wordy, enhance if vague.")
    improve_clicked = st.button("✨ Improve", use_container_width=True)

    if improve_clicked:
        refine_mode = _auto_refine_mode(analysis)
        strategy_label = "⚡ Simplifying (prompt is long and specific)…" if refine_mode == "simplify" else "🔬 Enhancing (prompt needs more detail)…"
        with st.spinner(strategy_label):
            preferred_ana = st.session_state.get("s_analyzer_model")
            if preferred_ana == "(auto)":
                preferred_ana = None
            refinement = refine_prompt(user_prompt, analysis, timeout=timeout_sec, mode=refine_mode, preferred_analyzer=preferred_ana)
        if refinement["error"]:
            err_info = interpret_error(refinement["error"], refinement.get("refiner_model", "Refiner"))
            st.error(f"{err_info['icon']} **{err_info['title']}** — {err_info['message']}")
            st.caption(f"💡 {err_info['hint']}")
        else:
            refined_text = refinement["refined"]
            new_analysis = analyse_prompt(refined_text)
            mode_label = "simplified" if refine_mode == "simplify" else "enhanced"
            st.session_state.refinement_history.append({"version": f"v{len(st.session_state.refinement_history) + 1} ({mode_label})", "prompt": refined_text, "quality": new_analysis["quality"], "changes": refinement["changes"]})
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
                if user_prompt.strip():
                    original_analysis = analyse_prompt(user_prompt)
                    if not st.session_state.query_history or st.session_state.query_history[-1]["full_prompt"] != user_prompt:
                        _ref_entry = {
                            "timestamp": datetime.now().strftime("%d.%m.%Y · %H:%M:%S"),
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
                        }
                        st.session_state.query_history.append(_ref_entry)
                        save_history_entry(_user_id, _ref_entry)
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
    st.info("👆 When you're happy, click **📡 Broadcast to selected Models** to send it out.")


# ── STAGE 1: Analyze (stores result, renders below broadcast) ──
if analyze_clicked and user_prompt.strip():
    st.session_state.last_analysis = analyse_prompt(user_prompt)
    st.session_state._pending_refined = None
    st.session_state._pending_analysis = None
    if not st.session_state.refinement_history:
        st.session_state.refinement_history.append({"version": "v1 (original)", "prompt": user_prompt, "quality": st.session_state.last_analysis["quality"], "changes": []})

# ── STAGE 2: Broadcast — compute & store, rendering happens below ──
if broadcast_clicked and user_prompt.strip():
    st.session_state.broadcast_output = None  # clear while running
    st.session_state._loaded_history_entry = None
    analysis = analyse_prompt(user_prompt)
    models_to_call = selected_models

    effective_prompt = build_effective_prompt(user_prompt)

    n_models = len(models_to_call)
    progress_bar = st.progress(0, text=f"📡 Calling {n_models} model(s) in parallel…")
    status_area = st.empty()

    results = []
    with ThreadPoolExecutor(max_workers=n_models) as pool:
        futures = {
            pool.submit(call_model, name, effective_prompt, max_tokens, temperature, timeout_sec): name
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

    order = {name: i for i, name in enumerate(models_to_call)}
    results.sort(key=lambda r: order.get(r["name"], 99))

    progress_bar.progress(1.0, text=f"✅ All {n_models} model(s) finished")
    time.sleep(0.5)
    progress_bar.empty()
    status_area.empty()

    is_arena = "Arena" in mode
    synthesis = None

    if not is_arena:
        preferred_agg = st.session_state.get("s_aggregator_model")
        if preferred_agg == "(auto)":
            preferred_agg = None
        with st.spinner("🧠 Aggregation Engine: merging outputs…"):
            synthesis = aggregate_responses(effective_prompt, results, preferred_aggregator=preferred_agg)

    # ── Store to session_state so output persists across reruns ──
    st.session_state.broadcast_output = {
        "synthesis": synthesis,
        "results": results,
        "mode": "Arena" if is_arena else "Aggregator",
        "prompt": user_prompt,
        "quality": analysis["quality"],
        "complexity": analysis["complexity_level"],
        "n_models": len(models_to_call),
        "has_timing": True,
        "from_history": False,
    }

    # ── Save run to Prompt Log (query_history) ──
    _summary_text = summarise_prompt(user_prompt)
    history_entry = {
        "timestamp":        datetime.now().strftime("%d.%m.%Y · %H:%M:%S"),
        "summary":          _summary_text,
        "prompt":           user_prompt[:120] + ("…" if len(user_prompt) > 120 else ""),
        "full_prompt":      user_prompt,
        "mode":             "Arena" if is_arena else "Aggregator",
        "quality":          analysis["quality"],
        "complexity":       analysis["complexity_level"],
        "analysis":         analysis,
        "models_used":      [r["name"] for r in results],
        "successful":       [r["name"] for r in results if r["response"]],
        "failed":           [r["name"] for r in results if r["error"]],
        "times":            {r["name"]: r["time"] for r in results},
        "word_counts":      {r["name"]: r["words"] for r in results},
        "responses":        {r["name"]: r["response"] for r in results if r["response"]},
        "errors":           {r["name"]: r["error"] for r in results if r["error"]},
        "synthesis":        synthesis if not is_arena else None,
        "aggregator_model": st.session_state.get("s_aggregator_model"),
        "analyzer_model":   st.session_state.get("s_analyzer_model"),
        "output_models":    list(models_to_call),
    }
    if not st.session_state.query_history or st.session_state.query_history[-1]["full_prompt"] != user_prompt:
        st.session_state.query_history.append(history_entry)
        save_history_entry(_user_id, history_entry)


# ── Persistent results render (live broadcast OR history recall) ──
_bo = st.session_state.get("broadcast_output")
if _bo:
    _bo_mode = _bo.get("mode", "Aggregator")
    _bo_synthesis = _bo.get("synthesis")
    _bo_results = _bo.get("results") or []
    _bo_has_timing = _bo.get("has_timing", False)
    _bo_from_history = _bo.get("from_history", False)

    st.markdown("---")
    st.markdown("### 📡 Broadcasting")
    _bo_quality = _bo.get("quality", 0)
    _bo_complexity = _bo.get("complexity", "")
    _bo_n = _bo.get("n_models", len(_bo_results))
    st.caption(f"Final prompt quality: **{_bo_quality}/100** · complexity: **{_bo_complexity}** · {_bo_n} model(s)")

    if _bo_from_history:
        _bo_ts = _bo.get("timestamp", "")
        _bo_summary = _bo.get("summary", "")
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid #1e3a5f;border-left:4px solid #3b82f6;'
            f'border-radius:8px;padding:0.6rem 1rem;margin:0.4rem 0 0.8rem;">'
            f'<span style="color:#93c5fd;font-size:0.82rem;font-weight:600">📜 Loaded from Prompt Log</span>'
            f'<span style="color:#475569;font-size:0.78rem;margin-left:0.8rem">{_bo_ts} · {_bo_summary}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if _bo_mode == "Arena":
        st.markdown("#### 🏟️ Arena — Side-by-Side Comparison")
        _successful = [r for r in _bo_results if r.get("response")]
        if _successful and _bo_has_timing:
            _fastest = min(_successful, key=lambda r: r["time"])
            _longest = max(_successful, key=lambda r: r["words"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("⚡ Fastest", _fastest["name"], f"{_fastest['time']}s")
            mc2.metric("📝 Most Detailed", _longest["name"], f"{_longest['words']} words")
            mc3.metric("✅ Responses", f"{len(_successful)}/{len(_bo_results)}")
        if _bo_results:
            _cols = st.columns(max(1, len(_bo_results)))
            for _i, _r in enumerate(_bo_results):
                _cfg = MODELS.get(_r["name"], {})
                with _cols[_i]:
                    if _bo_has_timing:
                        st.markdown(f'<div class="model-card"><div class="model-header"><span class="model-name" style="color:{_cfg.get("color","#94a3b8")}">{_cfg.get("icon","🤖")} {_r["name"]}</span><span class="model-meta">{_r["time"]}s</span></div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'**{_cfg.get("icon","🤖")} {_r["name"]}**')
                    if _r.get("error"):
                        _ei = interpret_error(_r["error"], _r["name"])
                        st.error(f"{_ei['icon']} **{_ei['title']}**\n\n{_ei['message']}")
                        st.caption(f"💡 {_ei['hint']}")
                        if _ei.get("raw"):
                            with st.expander("🔍 Raw error"):
                                st.code(_ei["raw"], language="text")
                    else:
                        if _bo_has_timing:
                            st.markdown(f'<span class="metric-pill">⏱ {_r["time"]}s</span><span class="metric-pill">📝 {_r["words"]} words</span>', unsafe_allow_html=True)
                            st.markdown("")
                        st.markdown(_r.get("response", ""))
    else:
        # ── Aggregator ──
        st.markdown("#### 🧠 Aggregator — Synthesised Output")
        if _bo_synthesis:
            _rec_answer, _supporting = _extract_recommended_answer(_bo_synthesis)
            _src_names = [r["name"] for r in _bo_results if r.get("response")]
            if _rec_answer:
                st.markdown("### 📌 Recommended Answer")
                st.markdown(_rec_answer)
                if _supporting:
                    st.markdown("---")
                    st.markdown(_supporting)
                st.markdown("---")
            else:
                st.markdown("### 📌 Final Synthesised Answer")
                st.markdown(_bo_synthesis)
                st.markdown("---")
            if _src_names:
                st.caption(f"Sources: {', '.join(_src_names)}")
            st.download_button("📥 Download Result", data=_bo_synthesis, file_name="quorum_result.md", mime="text/markdown", key="dl_bo_synthesis")
        else:
            st.warning("Not enough successful responses to aggregate (need at least 2).")

        if _bo_results:
            st.markdown("#### 📋 Individual Model Responses")
            _ind_cols = st.columns(max(1, len(_bo_results)))
            for _i, _r in enumerate(_bo_results):
                _cfg = MODELS.get(_r["name"], {})
                with _ind_cols[_i]:
                    if _r.get("response"):
                        st.markdown(f"**{_cfg.get('icon','🤖')} {_r['name']}**")
                        if _bo_has_timing:
                            st.caption(f"*{_r['time']}s · {_r['words']} words*")
                        st.markdown(_r["response"])
                    elif _r.get("error"):
                        _ei = interpret_error(_r["error"], _r["name"])
                        st.warning(f"{_ei['icon']} **{_r['name']}**\n\n{_ei['title']}: {_ei['message']}")
                        if _ei.get("raw"):
                            with st.expander("🔍 Raw error"):
                                st.code(_ei["raw"], language="text")

    if _bo_has_timing and _bo_results:
        with st.expander("📊 Response Time Comparison"):
            _max_t = max(r["time"] for r in _bo_results) if _bo_results else 1
            for _r in _bo_results:
                _cfg = MODELS.get(_r["name"], {})
                _bp = _r["time"] / _max_t if _max_t > 0 else 0
                _lbl = "✅" if not _r.get("error") else "❌"
                st.markdown(f"**{_cfg.get('icon','🤖')} {_r['name']}** — {_r['time']}s  ·  {_r['words']} words {_lbl}")
                st.progress(min(_bp, 1.0))

# ── Prompt Diagnostic (rendered AFTER broadcast results) ──
analysis = st.session_state.last_analysis
if analysis and show_prompt_analysis:
    _render_diagnostic_section(analysis, user_prompt)

elif (analyze_clicked or broadcast_clicked) and not user_prompt.strip():
    st.warning("Please enter a prompt first.")

# ──────────────────────────────────────────────
# 7 · QUERY HISTORY  (moved to sidebar — see 💬 History section)
# ──────────────────────────────────────────────
