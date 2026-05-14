"""
Tests for the PromptAnalyzer class.

These tests guard against the five failure modes identified during the
v14 Layer-1 review. If any of them fail, it means the scoring heuristic
has regressed back to a known-bad behaviour.
"""
import pytest


class TestPromptAnalyzerBasics:
    """Shape and contract tests, does the analyzer return what callers expect?"""

    def test_analyse_returns_expected_keys(self, app_module):
        result = app_module["PromptAnalyzer"]("Write a summary.").analyse()
        for key in ("quality", "complexity", "complexity_level", "category",
                    "suggestions", "dimensions", "word_count", "sentence_count"):
            assert key in result, f"Missing key: {key}"

    def test_seven_dimensions_returned(self, app_module):
        dims = app_module["PromptAnalyzer"]("Write a summary.").analyse()["dimensions"]
        expected = {"Explicit Direction", "Defined Output", "Context & Motivation",
                    "Specificity", "Uncertainty Safeguard", "Chain of Thought",
                    "Positive Framing"}
        assert set(dims.keys()) == expected

    def test_each_dimension_has_score_evidence_tip(self, app_module):
        dims = app_module["PromptAnalyzer"]("Write a summary.").analyse()["dimensions"]
        for name, d in dims.items():
            assert "score" in d and 0 <= d["score"] <= 100, f"{name} has bad score"
            assert "evidence" in d and isinstance(d["evidence"], list), f"{name} missing evidence list"
            assert "tip" in d, f"{name} missing tip field"

    def test_module_wrapper_matches_class(self, app_module):
        """analyse_prompt() must return identical output to PromptAnalyzer().analyse()."""
        text = "Compare A and B."
        assert app_module["analyse_prompt"](text) == app_module["PromptAnalyzer"](text).analyse()


class TestComplexityScoring:
    """Layer-1 Fix 2: complexity should be driven by task signals, not word count."""

    def test_short_reasoning_task_scores_complex(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Compare Python and Rust for systems programming — discuss the trade-offs."
        ).analyse()
        assert result["complexity"] >= 50, (
            f"A reasoning-heavy compare/trade-off prompt should score Moderate or Complex, "
            f"got {result['complexity']}"
        )

    def test_long_trivial_task_scores_simple(self, app_module):
        # Repetitive trivial text — high word count but no real task signal
        result = app_module["PromptAnalyzer"](
            "Please count the words in this paragraph. " * 8
        ).analyse()
        assert result["complexity"] < 50, (
            f"A verbose-but-trivial prompt should not score Complex, got {result['complexity']}"
        )

    def test_haiku_stays_simple(self, app_module):
        result = app_module["PromptAnalyzer"]("Write a haiku about the ocean.").analyse()
        assert result["complexity_level"] == "Simple"


class TestDefinedOutputScoring:
    """Layer-1 Fix 1: 'JSON' alone is not enough — must have a schema."""

    def test_bare_format_word_does_not_max_score(self, app_module):
        result = app_module["PromptAnalyzer"]("Write me a JSON").analyse()
        # Bare format mention should now score around 25, not the old 40+
        assert result["dimensions"]["Defined Output"]["score"] < 40

    def test_format_with_real_schema_scores_high(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Analyze the user data and return JSON with these fields:\n"
            "- name: string\n- age: integer\n- email: string"
        ).analyse()
        assert result["dimensions"]["Defined Output"]["score"] >= 50


class TestSpecificityScoring:
    """Layer-1 Fix 4: only constraint-position entities count, not topic mentions."""

    def test_bare_topic_mentions_do_not_inflate_specificity(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Tell me about Python and API and JSON and Anthropic and React"
        ).analyse()
        # Specificity should be low because no entity acts as a constraint
        assert result["dimensions"]["Specificity"]["score"] <= 30

    def test_entities_as_constraints_are_rewarded(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Write a script using Python with FastAPI for a REST endpoint targeting Postgres."
        ).analyse()
        assert result["dimensions"]["Specificity"]["score"] >= 40


class TestChainOfThoughtScoring:
    """Layer-1 Fix 3: CoT triggers on reasoning signals, not on word count."""

    def test_long_creative_does_not_trigger_cot(self, app_module):
        prompt = (
            "Write a detailed blog post describing a relaxing weekend in the mountains, "
            "including the scenery, the food, the lodge, the activities, and the people "
            "you might meet, in a warm conversational tone aimed at a travel magazine audience."
        )
        result = app_module["PromptAnalyzer"](prompt).analyse()
        assert result["dimensions"]["Chain of Thought"]["score"] == 100, (
            "Long creative writing should not be penalised for lacking CoT"
        )

    def test_short_reasoning_triggers_cot(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Compare merge sort and quicksort, weigh the trade-offs."
        ).analyse()
        assert result["dimensions"]["Chain of Thought"]["score"] < 100


class TestPositiveFramingScoring:
    """Layer-1 Fix 5: negatives are only bad when there's no positive direction."""

    def test_mixed_framing_is_not_penalised_heavily(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Write a Python function using FastAPI that returns a JSON response. "
            "Don't use globals, never use eval, and avoid blocking calls."
        ).analyse()
        # Positive direction + negatives should score >= 70
        assert result["dimensions"]["Positive Framing"]["score"] >= 70

    def test_pure_negatives_are_still_penalised(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Don't use bullet points. Never use markdown. Avoid emojis."
        ).analyse()
        # No positive direction → should still be flagged
        assert result["dimensions"]["Positive Framing"]["score"] < 70


class TestCategorisation:
    """The category field should map to task type."""

    def test_code_prompt_categorised_as_code(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Write a Python function that sorts a list."
        ).analyse()
        assert "Code" in result["category"]

    def test_creative_prompt_categorised_as_creative(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Write a short story about a lonely lighthouse keeper."
        ).analyse()
        assert "Creative" in result["category"]

    def test_analysis_prompt_categorised_as_analysis(self, app_module):
        result = app_module["PromptAnalyzer"](
            "Compare and contrast these two policy positions."
        ).analyse()
        assert "Analysis" in result["category"]

    def test_substring_keyword_collision_is_a_known_bug(self, app_module):
        """
        The category keyword lookup uses substring matching ('api' in self.lower),
        so 'capitalism' incorrectly matches the 'api' keyword from the code-task list.

        This test documents the current (buggy) behaviour so a future fix
        will visibly break the test and force a reviewer's attention.
        When the bug is fixed (e.g. by switching to whole-word matching),
        flip this assertion to `"Analysis"` and remove the bug-comment.
        """
        result = app_module["PromptAnalyzer"](
            "Compare and contrast capitalism and socialism."
        ).analyse()
        # 'capitalism' contains 'api' as a substring → false positive code match
        assert "Code" in result["category"], (
            "Bug fixed? Update this test to assert 'Analysis' instead."
        )
