"""
Unit tests for data_processor.py

Tests:
- _star_to_score:   converts star rating to -1 to +1 score
- extract_keywords: extracts top N keywords from data items
- _analyse_item:    scores individual review/news items
- analyse_business: aggregates scores across all items
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import (
    _star_to_score, extract_keywords, _analyse_item, analyse_business,
    extract_keyword_split, extractive_summary, _get_category_weights,
)

pytestmark = pytest.mark.unit

# ── _star_to_score ────────────────────────────────────────────────────────────

class TestStarToScore:
    def test_5_star_returns_1(self):
        assert _star_to_score(5) == 1.0

    def test_1_star_returns_minus_1(self):
        assert _star_to_score(1) == -1.0

    def test_3_star_returns_0(self):
        assert _star_to_score(3) == 0.0

    def test_4_star_returns_0_5(self):
        assert _star_to_score(4) == 0.5

    def test_2_star_returns_minus_0_5(self):
        assert _star_to_score(2) == -0.5

    def test_invalid_returns_none(self):
        assert _star_to_score(None) is None

    def test_string_number_converts(self):
        assert _star_to_score("4") == 0.5

    def test_non_numeric_string_returns_none(self):
        assert _star_to_score("abc") is None

# ── extract_keywords ──────────────────────────────────────────────────────────

class TestExtractKeywords:
    def test_returns_list(self):
        data = [{"body": "The food was great and the service was excellent"}]
        result = extract_keywords(data)
        assert isinstance(result, list)

    def test_returns_at_most_top_n(self):
        data = [{"body": "great food service staff clean fast friendly good nice"}]
        result = extract_keywords(data, top_n=3)
        assert len(result) <= 3

    def test_excludes_business_name_words(self):
        data = [{"body": "subway has great sandwiches and subway is fast"}]
        result = extract_keywords(data, business_name="Subway")
        assert "subway" not in result

    def test_excludes_stop_words(self):
        data = [{"body": "the food was really very good and the service was also great"}]
        result = extract_keywords(data)
        assert "the" not in result
        assert "and" not in result

    def test_empty_data_returns_empty_list(self):
        result = extract_keywords([])
        assert result == []

    def test_items_with_no_body_skipped(self):
        data = [{"body": ""}, {"body": None}]
        result = extract_keywords(data)
        assert result == []

# ── _analyse_item ─────────────────────────────────────────────────────────────

class TestAnalyseItem:
    def test_positive_review_returns_positive_score(self):
        item = {"source": "google_maps_reviews", "body": "Amazing food and excellent service!", "rating": 5}
        score = _analyse_item(item)
        assert score is not None
        assert score > 0

    def test_negative_review_returns_negative_score(self):
        item = {"source": "google_maps_reviews", "body": "Terrible experience, disgusting food.", "rating": 1}
        score = _analyse_item(item)
        assert score is not None
        assert score < 0

    def test_rating_only_no_body(self):
        item = {"source": "google_maps_reviews", "body": "", "rating": 5}
        score = _analyse_item(item)
        assert score == 1.0

    def test_no_body_no_rating_returns_none(self):
        item = {"source": "google_maps_reviews", "body": "", "rating": None}
        score = _analyse_item(item)
        assert score is None

    def test_news_item_returns_score(self):
        item = {"source": "newsapi", "title": "Company reports record profits", "body": "The firm posted record earnings beating analyst expectations."}
        score = _analyse_item(item)
        assert score is not None
        assert -1.0 <= score <= 1.0

    def test_news_too_short_returns_none(self):
        item = {"source": "newsapi", "title": "News", "body": "Ok"}
        score = _analyse_item(item)
        assert score is None

    def test_score_within_bounds(self):
        item = {"source": "google_maps_reviews", "body": "Good place overall", "rating": 4}
        score = _analyse_item(item)
        assert -1.0 <= score <= 1.0

    def test_review_field_used_as_body(self):
        item = {"source": "google_maps_reviews", "review": "Fantastic experience!", "rating": 5}
        score = _analyse_item(item)
        assert score is not None

# ── analyse_business ──────────────────────────────────────────────────────────

class TestAnalyseBusiness:
    def test_returns_required_keys(self):
        data = [{"source": "google_maps_reviews", "body": "Great food!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        for key in ["business_name", "location", "category", "overall_sentiment",
                    "overall_rating", "overall_score", "items_analysed", "keywords", "breakdown"]:
            assert key in result

    def test_empty_data_returns_defaults(self):
        result = analyse_business("TestCafe", "Sydney", "cafe", [])
        assert result["items_analysed"] == 0
        assert result["overall_sentiment"] == "neutral"
        assert result["overall_rating"] == 3
        assert result["overall_score"] == 0.0

    def test_positive_reviews_give_positive_sentiment(self):
        data = [
            {"source": "google_maps_reviews", "body": "Amazing! Best place ever!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Incredible food and service!", "rating": 5},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["overall_sentiment"] == "positive"
        assert result["overall_score"] > 0

    def test_negative_reviews_give_negative_sentiment(self):
        data = [
            {"source": "google_maps_reviews", "body": "Awful, terrible, disgusting place.", "rating": 1},
            {"source": "google_maps_reviews", "body": "Worst experience ever, horrible service.", "rating": 1},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["overall_sentiment"] == "negative"
        assert result["overall_score"] < 0

    def test_items_analysed_count_is_correct(self):
        data = [
            {"source": "google_maps_reviews", "body": "Great!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Good!", "rating": 4},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["items_analysed"] == 2

    def test_business_metadata_preserved(self):
        result = analyse_business("Subway", "Melbourne", "restaurant", [])
        assert result["business_name"] == "Subway"
        assert result["location"] == "Melbourne"
        assert result["category"] == "restaurant"

    def test_breakdown_contains_per_item_scores(self):
        data = [{"source": "google_maps_reviews", "body": "Great food!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert len(result["breakdown"]) == 1
        item = result["breakdown"][0]
        assert "sentiment" in item
        assert "score" in item

    # ── New fields present ────────────────────────────────────────────────────

    def test_returns_keyword_split(self):
        data = [
            {"source": "google_maps_reviews", "body": "Amazing food and excellent service!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Horrible terrible awful experience.", "rating": 1},
            {"source": "google_maps_reviews", "body": "Amazing food and excellent service!", "rating": 5},
            {"source": "google_maps_reviews", "body": "Horrible terrible awful experience.", "rating": 1},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert "keyword_split" in result
        assert "positive" in result["keyword_split"]
        assert "negative" in result["keyword_split"]

    def test_returns_summary(self):
        data = [
            {"source": "google_maps_reviews", "body": "The food was absolutely wonderful and the staff were incredibly attentive and helpful.", "rating": 5},
            {"source": "newsapi", "title": "Local cafe wins award", "body": "The cafe has won a regional award for its outstanding service and quality food offerings."},
        ]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert "summary" in result
        assert isinstance(result["summary"], str)

    def test_returns_explanation(self):
        data = [{"source": "google_maps_reviews", "body": "Amazing food!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert "explanation" in result
        assert len(result["explanation"]) > 10

    def test_returns_incremental_flag(self):
        data = [{"source": "google_maps_reviews", "body": "Great!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert "incremental" in result
        assert result["incremental"] is False

# ── extract_keyword_split ─────────────────────────────────────────────────────

class TestExtractKeywordSplit:
    def _make_scored(self, items_scores):
        """Helper: build (item, score) pairs directly."""
        return [({"source": "google_maps_reviews", "body": text, "rating": None}, score)
                for text, score in items_scores]

    def test_returns_positive_and_negative_keys(self):
        scored = self._make_scored([
            ("amazing excellent wonderful service", 0.8),
            ("amazing excellent wonderful service", 0.8),
            ("terrible awful horrible disaster", -0.8),
            ("terrible awful horrible disaster", -0.8),
        ])
        result = extract_keyword_split(scored)
        assert "positive" in result
        assert "negative" in result

    def test_positive_words_in_positive_items(self):
        scored = self._make_scored([
            ("amazing excellent wonderful food", 0.8),
            ("amazing excellent wonderful food", 0.8),
        ])
        result = extract_keyword_split(scored)
        # Words from positive items should appear in positive list
        assert len(result["positive"]) > 0

    def test_negative_words_in_negative_items(self):
        scored = self._make_scored([
            ("terrible awful horrible experience", -0.8),
            ("terrible awful horrible experience", -0.8),
        ])
        result = extract_keyword_split(scored)
        assert len(result["negative"]) > 0

    def test_business_name_excluded(self):
        scored = self._make_scored([
            ("subway sandwiches are amazing and great", 0.8),
            ("subway sandwiches are amazing and great", 0.8),
        ])
        result = extract_keyword_split(scored, business_name="Subway")
        assert "subway" not in result["positive"]
        assert "subway" not in result["negative"]

    def test_empty_scored_returns_empty_lists(self):
        result = extract_keyword_split([])
        assert result == {"positive": [], "negative": []}

    def test_respects_top_n(self):
        scored = self._make_scored([
            ("great excellent amazing wonderful brilliant fantastic food", 0.9),
            ("great excellent amazing wonderful brilliant fantastic food", 0.9),
        ])
        result = extract_keyword_split(scored, top_n=2)
        assert len(result["positive"]) <= 2

# ── extractive_summary ────────────────────────────────────────────────────────

class TestExtractiveSummary:
    def test_returns_string(self):
        data = [{"body": "The food was great and the staff were very friendly and helpful."}]
        assert isinstance(extractive_summary(data), str)

    def test_empty_data_returns_empty_string(self):
        assert extractive_summary([]) == ""

    def test_short_data_returns_all_sentences(self):
        data = [{"body": "The food was great and the staff were very friendly and helpful."}]
        result = extractive_summary(data, n_sentences=3)
        assert len(result) > 0

    def test_long_data_returns_at_most_n_sentences(self):
        sentences = [f"Sentence number {i} about the business with various descriptive words here." for i in range(20)]
        data = [{"body": " ".join(sentences)}]
        result = extractive_summary(data, n_sentences=3)
        # Count rough sentence boundaries
        count = result.count(". ") + result.count(".")
        assert count <= 5  # generous bound

    def test_uses_both_body_and_title(self):
        data = [{"title": "Great restaurant wins award for amazing service quality.", "body": ""}]
        result = extractive_summary(data)
        assert isinstance(result, str)

    def test_filters_very_short_sentences(self):
        data = [{"body": "Ok. Fine. The food was absolutely outstanding with perfect service and ambiance."}]
        result = extractive_summary(data, n_sentences=1)
        # Should pick the long sentence, not "Ok" or "Fine"
        assert len(result.split()) > 5

# ── Category weights ──────────────────────────────────────────────────────────

class TestCategoryWeights:
    def test_restaurant_weights_reviews_heavily(self):
        w = _get_category_weights("restaurant")
        assert w["review"] > w["news"]

    def test_finance_weights_news_heavily(self):
        w = _get_category_weights("finance")
        assert w["news"] > w["review"]

    def test_unknown_category_returns_default(self):
        w = _get_category_weights("taxidermy")
        assert w["news"] == 0.5
        assert w["review"] == 0.5

    def test_partial_match_works(self):
        # "fast food" should match the "food" key
        w = _get_category_weights("fast food")
        assert w["review"] > w["news"]

    def test_category_weight_applied_to_score(self):
        # Finance category: same data, news should dominate
        news_item = {"source": "newsapi", "title": "Record profits and massive growth", "body": "The company posted record profits beating all analyst expectations with massive revenue growth."}
        review_item = {"source": "google_maps_reviews", "body": "Worst bank ever. Terrible service.", "rating": 1}
        data = [news_item, review_item]

        finance_result    = analyse_business("TestBank", "Sydney", "finance", data)
        restaurant_result = analyse_business("TestBank", "Sydney", "restaurant", data)

        # Finance weights news (positive) more → should score higher than restaurant (weights review/negative more)
        assert finance_result["overall_score"] > restaurant_result["overall_score"]

# ── Incremental scoring ───────────────────────────────────────────────────────

class TestIncrementalScoring:
    def test_incremental_false_without_prev_score(self):
        data = [{"source": "google_maps_reviews", "body": "Great place!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data)
        assert result["incremental"] is False

    def test_incremental_true_with_prev_score(self):
        data = [{"source": "google_maps_reviews", "body": "Great place!", "rating": 5}]
        result = analyse_business("TestCafe", "Sydney", "cafe", data, prev_score=0.0)
        assert result["incremental"] is True

    def test_prev_score_pulls_result_toward_it(self):
        data = [{"source": "google_maps_reviews", "body": "Amazing absolutely perfect wonderful!", "rating": 5}]
        raw    = analyse_business("TestCafe", "Sydney", "cafe", data)
        pulled = analyse_business("TestCafe", "Sydney", "cafe", data, prev_score=-1.0)
        # prev_score=-1.0 (very negative) should pull the score down vs no blending
        assert pulled["overall_score"] < raw["overall_score"]

    def test_prev_score_is_blended_not_replaced(self):
        data = [{"source": "google_maps_reviews", "body": "Amazing absolutely perfect wonderful!", "rating": 5}]
        pulled = analyse_business("TestCafe", "Sydney", "cafe", data, prev_score=-1.0)
        # Result should NOT be -1.0 (it's blended 70% new)
        assert pulled["overall_score"] > -0.5

    def test_empty_data_with_prev_score_returns_zero(self):
        # No items means no score — incremental flag off, prev_score unused
        result = analyse_business("TestCafe", "Sydney", "cafe", [], prev_score=0.9)
        assert result["overall_score"] == 0.0
        assert result["incremental"] is False
