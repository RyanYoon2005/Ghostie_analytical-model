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

pytestmark = pytest.mark.unit

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

# ── Negation handling ─────────────────────────────────────────────────────────

class TestNegationHandling:
    def test_not_profit_scores_lower_than_profit(self):
        positive = _lexicon_score("The company reported profit this quarter.")
        negated  = _lexicon_score("The company did not report profit this quarter.")
        assert negated < positive

    def test_not_loss_scores_higher_than_loss(self):
        negative = _lexicon_score("The company reported a loss.")
        negated  = _lexicon_score("The company did not report a loss.")
        assert negated > negative

    def test_no_growth_scores_lower_than_growth(self):
        positive = _lexicon_score("There was strong growth in revenue.")
        negated  = _lexicon_score("There was no growth in revenue.")
        assert negated < positive

    def test_never_profit_scores_lower_than_profit(self):
        positive = _lexicon_score("The company reported profit every quarter.")
        negated  = _lexicon_score("The company never reported profit.")
        assert negated < positive

    def test_negation_window_expires_after_5_words(self):
        # "not" is 6 words before "profit" — window (5) should have expired
        score = _lexicon_score("The results were not what analysts had previously expected as profit rose.")
        assert score > 0

    def test_double_negation_restores_positive(self):
        # "not a loss" — double negative should push score back toward positive
        double_neg = _lexicon_score("This is not a loss for the company.")
        raw_neg    = _lexicon_score("This is a loss for the company.")
        assert double_neg > raw_neg

    def test_contraction_negation_works(self):
        # "doesn't" should be recognised as a negation word
        score = _lexicon_score("The company doesn't report losses.")
        # Negating "losses" (negative) → pushes score positive
        assert score > 0
