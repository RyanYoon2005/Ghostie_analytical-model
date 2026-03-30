"""
Unit tests for custom_analyser.py

Tests the financial lexicon and VADER blending:
- _lexicon_score: raw financial lexicon scoring
- custom_score:   combined VADER + lexicon score
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_analyser import _lexicon_score, custom_score

# ── _lexicon_score ────────────────────────────────────────────────────────────

class TestLexiconScore:
    def test_positive_financial_words_score_positive(self):
        score = _lexicon_score("The company reported strong profit growth and record revenue.")
        assert score > 0

    def test_negative_financial_words_score_negative(self):
        score = _lexicon_score("The company reported massive losses and filed for bankruptcy.")
        assert score < 0

    def test_neutral_text_scores_near_zero(self):
        score = _lexicon_score("The meeting was held on Monday.")
        assert abs(score) < 0.2

    def test_score_within_bounds(self):
        score = _lexicon_score("Profit surge, record earnings, strong growth, beat expectations.")
        assert -1.0 <= score <= 1.0

    def test_positive_phrase_detected(self):
        score = _lexicon_score("The company beat expectations this quarter.")
        assert score > 0

    def test_negative_phrase_detected(self):
        score = _lexicon_score("Results were below expectations and missed analyst targets.")
        assert score < 0

    def test_empty_text_returns_zero(self):
        score = _lexicon_score("")
        assert score == 0.0

# ── custom_score ──────────────────────────────────────────────────────────────

class TestCustomScore:
    def test_positive_review_scores_positive(self):
        score = custom_score("Amazing food, great service, absolutely wonderful experience!")
        assert score > 0

    def test_negative_review_scores_negative(self):
        score = custom_score("Terrible service, disgusting food, complete waste of money.")
        assert score < 0

    def test_score_within_bounds(self):
        score = custom_score("The quarterly results exceeded all analyst forecasts.")
        assert -1.0 <= score <= 1.0

    def test_financial_positive_blends_correctly(self):
        score = custom_score("Record profits and strong revenue growth beat all expectations.")
        assert score > 0.1

    def test_financial_negative_blends_correctly(self):
        score = custom_score("Heavy losses and bankruptcy filing dragged down the stock.")
        assert score < -0.1

    def test_empty_text_returns_float(self):
        score = custom_score("")
        assert isinstance(score, float)
