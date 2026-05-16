import re
import json
import os
from collections import Counter, defaultdict
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
from nltk.corpus import stopwords
from analyser import analyse, combined_rating, combined_label

_LEARNED_LEXICON_PATH = os.environ.get(
    "LEARNED_LEXICON_PATH",
    os.path.join(os.path.dirname(__file__), "learned_lexicon.json")
)

# ── Category-specific source weights ──────────────────────────────────────────
# news_weight + review_weight should sum to 1.0
CATEGORY_WEIGHTS = {
    "restaurant":  {"news": 0.2, "review": 0.8},
    "food":        {"news": 0.2, "review": 0.8},
    "cafe":        {"news": 0.2, "review": 0.8},
    "hospitality": {"news": 0.3, "review": 0.7},
    "retail":      {"news": 0.3, "review": 0.7},
    "finance":     {"news": 0.7, "review": 0.3},
    "banking":     {"news": 0.7, "review": 0.3},
    "insurance":   {"news": 0.7, "review": 0.3},
    "technology":  {"news": 0.5, "review": 0.5},
    "tech":        {"news": 0.5, "review": 0.5},
    "software":    {"news": 0.5, "review": 0.5},
    "healthcare":  {"news": 0.6, "review": 0.4},
    "medical":     {"news": 0.6, "review": 0.4},
    "default":     {"news": 0.5, "review": 0.5},
}

def _get_category_weights(category: str) -> dict:
    cat = category.lower()
    for key, weights in CATEGORY_WEIGHTS.items():
        if key != "default" and key in cat:
            return weights
    return CATEGORY_WEIGHTS["default"]

# ── Learned lexicon update ─────────────────────────────────────────────────────

def _update_learned_lexicon(keywords: list[str], data: list) -> None:
    from custom_analyser import POSITIVE_WORDS, NEGATIVE_WORDS
    hardcoded = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS)

    word_scores: dict[str, list[float]] = defaultdict(list)
    for item in data:
        score = _analyse_item(item)
        if score is None:
            continue
        text = f"{item.get('title', '')} {item.get('body') or item.get('review', '')}".lower()
        for kw in keywords:
            if kw in text:
                word_scores[kw].append(score)

    if os.path.exists(_LEARNED_LEXICON_PATH):
        with open(_LEARNED_LEXICON_PATH) as f:
            learned = json.load(f)
    else:
        learned = {"positive": {}, "negative": {}}

    updated = []
    for kw in keywords:
        if kw in hardcoded:
            continue
        scores = word_scores.get(kw, [])
        if len(scores) < 2:
            continue
        avg = sum(scores) / len(scores)
        if avg > 0.2:
            learned["positive"][kw] = round(min(avg, 0.8), 2)
            learned["negative"].pop(kw, None)
            updated.append(f"+{kw}")
        elif avg < -0.2:
            learned["negative"][kw] = round(min(abs(avg), 0.8), 2)
            learned["positive"].pop(kw, None)
            updated.append(f"-{kw}")

    with open(_LEARNED_LEXICON_PATH, "w") as f:
        json.dump(learned, f, indent=2)
    if updated:
        print(f"Learned lexicon updated: {', '.join(updated)}")

# ── Stop words ────────────────────────────────────────────────────────────────

_STOP_WORDS = set(stopwords.words('english')) | {
    "card", "cards", "gift", "swap", "brand", "brands", "option", "options",
    "also", "one", "two", "three", "may", "get", "use", "via", "per",
    "including", "available", "new", "like", "says", "said", "will", "year",
    "years", "month", "months", "week", "ago", "day", "days", "time",
    "company", "companies", "market", "report", "business", "product", "products",
}

# ── Keyword extraction ────────────────────────────────────────────────────────

def _build_stop(business_name: str, location: str, category: str) -> set:
    query_words = set()
    for phrase in (business_name, location, category):
        query_words.update(re.findall(r'\b[a-z]{3,}\b', phrase.lower()))
    return _STOP_WORDS | query_words

def extract_keywords(data: list, top_n: int = 5,
                     business_name: str = "", location: str = "", category: str = "") -> list[str]:
    stop = _build_stop(business_name, location, category)
    counts = Counter()
    for item in data:
        if item.get("source") == "newsapi":
            text = f"{item.get('title', '')} {item.get('body', '')}"
        else:
            text = item.get("body") or item.get("review") or ""
        if not text.strip():
            continue
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        counts.update(w for w in words if w not in stop)
    return [word for word, _ in counts.most_common(top_n)]

def extract_keyword_split(scored: list[tuple[dict, float]], top_n: int = 5,
                          business_name: str = "", location: str = "", category: str = "") -> dict:
    """Return top keywords split into positive and negative lists.

    Args:
        scored: list of (item, score) pairs already computed by _analyse_item.
    """
    stop = _build_stop(business_name, location, category)
    word_scores: dict[str, list[float]] = defaultdict(list)

    for item, score in scored:
        if item.get("source") == "newsapi":
            text = f"{item.get('title', '')} {item.get('body', '')}"
        else:
            text = item.get("body") or item.get("review") or ""
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        for w in words:
            if w not in stop:
                word_scores[w].append(score)

    scored_words = [
        (word, sum(s) / len(s), len(s))
        for word, s in word_scores.items()
        if len(s) >= 2
    ]
    positives = sorted([(w, a) for w, a, _ in scored_words if a > 0.15], key=lambda x: -x[1])
    negatives = sorted([(w, a) for w, a, _ in scored_words if a < -0.15], key=lambda x: x[1])

    return {
        "positive": [w for w, _ in positives[:top_n]],
        "negative": [w for w, _ in negatives[:top_n]],
    }

# ── Extractive summary ────────────────────────────────────────────────────────

def extractive_summary(data: list, n_sentences: int = 3) -> str:
    """Return the most representative sentences from all collected items."""
    sentences = []
    for item in data:
        text = f"{item.get('title', '')} {item.get('body') or item.get('review') or ''}"
        for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
            sent = sent.strip()
            if len(sent.split()) >= 8:
                sentences.append(sent)

    if not sentences:
        return ""
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf = vectorizer.fit_transform(sentences)
        centroid = tfidf.mean(axis=0)
        scores = cosine_similarity(tfidf, centroid).flatten()
        top_indices = sorted(scores.argsort()[-n_sentences:])
        return " ".join(sentences[i] for i in top_indices)
    except Exception:
        return " ".join(sentences[:n_sentences])

# ── Score explanation ─────────────────────────────────────────────────────────

def _explain_score(results: list, overall_score: float, keyword_split: dict) -> str:
    news = [r for r in results if r["source"] == "newsapi"]
    reviews = [r for r in results if r["source"] != "newsapi"]

    parts = []
    if news:
        pos = sum(1 for r in news if r["sentiment"] == "positive")
        neg = sum(1 for r in news if r["sentiment"] == "negative")
        parts.append(f"{len(news)} news articles ({pos} positive, {neg} negative)")
    if reviews:
        ratings = [float(r["rating"]) for r in reviews if r.get("rating") is not None]
        star_text = f" averaging {sum(ratings)/len(ratings):.1f}★" if ratings else ""
        pos = sum(1 for r in reviews if r["sentiment"] == "positive")
        neg = sum(1 for r in reviews if r["sentiment"] == "negative")
        parts.append(f"{len(reviews)} reviews ({pos} positive, {neg} negative{star_text})")

    explanation = ("Score based on " + " and ".join(parts) + ".") if parts else "Score based on available data."

    pos_kw = keyword_split.get("positive", [])[:3]
    neg_kw = keyword_split.get("negative", [])[:3]
    if pos_kw:
        explanation += f" Positive signals: {', '.join(pos_kw)}."
    if neg_kw:
        explanation += f" Negative signals: {', '.join(neg_kw)}."

    if overall_score > 0.5:
        explanation += " Overall sentiment is strongly positive."
    elif overall_score > 0.15:
        explanation += " Overall sentiment is mildly positive."
    elif overall_score < -0.5:
        explanation += " Overall sentiment is strongly negative."
    elif overall_score < -0.15:
        explanation += " Overall sentiment is mildly negative."
    else:
        explanation += " Overall sentiment is neutral."

    return explanation

# ── Item analyser ─────────────────────────────────────────────────────────────

def _star_to_score(rating) -> float:
    try:
        r = float(rating)
        return (r - 3) / 2
    except (TypeError, ValueError):
        return None

def _analyse_item(item: dict) -> float | None:
    source = item.get("source", "")
    title  = item.get("title", "") or ""
    body   = item.get("body") or item.get("review") or ""

    raw_rating = item.get("metadata", {}).get("rating") if item.get("metadata") else None
    if raw_rating is None:
        raw_rating = item.get("rating")

    is_review = source == "google_maps_reviews" or (raw_rating is not None and not title)

    if is_review:
        star_score = _star_to_score(raw_rating)
        if body.strip():
            _, _, _, ml_score, _, _ = analyse(body, use_ml=False)
            if star_score is not None:
                blend = ml_score * 0.6 + star_score * 0.4
                tolerance = 0.1 + abs(star_score) * 0.4
                return max(star_score - tolerance, min(star_score + tolerance, blend))
            return ml_score
        elif star_score is not None:
            return star_score
        return None
    else:
        text = f"{title}. {body}".strip(". ")
        if len(text.split()) < 4:
            return None
        _, _, _, ml_score, _, _ = analyse(text)
        return ml_score

# ── Main aggregation ──────────────────────────────────────────────────────────

def analyse_business(business_name: str, location: str, category: str,
                     data: list, prev_score: float | None = None) -> dict:
    """
    Run sentiment analysis on each item and aggregate into an overall result.

    Args:
        prev_score: Previous overall_score in -1 to +1 range for incremental
                    blending. Pass None to compute from scratch.
    """
    # Score all items once
    scored: list[tuple[dict, float]] = []
    for item in data:
        score = _analyse_item(item)
        if score is not None:
            scored.append((item, score))

    results = []
    for item, score in scored:
        results.append({
            "id":        item.get("id", ""),
            "source":    item.get("source", ""),
            "title":     item.get("title", ""),
            "body":      (item.get("body") or item.get("review") or "")[:200],
            "rating":    (item.get("metadata", {}).get("rating") if item.get("metadata") else None) or item.get("rating"),
            "sentiment": combined_label(score),
            "score":     round(score, 3),
        })

    if not results:
        return {
            "business_name":     business_name,
            "location":          location,
            "category":          category,
            "overall_sentiment": "neutral",
            "overall_rating":    3,
            "overall_score":     0.0,
            "items_analysed":    0,
            "keywords":          extract_keywords(data, business_name=business_name, location=location, category=category),
            "keyword_split":     {"positive": [], "negative": []},
            "summary":           "",
            "explanation":       "No data available to analyse.",
            "incremental":       False,
            "breakdown":         [],
        }

    # Category-weighted score: separate news from reviews then blend
    weights = _get_category_weights(category)
    news_scores   = [score for item, score in scored if item.get("source") == "newsapi"]
    review_scores = [score for item, score in scored if item.get("source") != "newsapi"]

    if news_scores and review_scores:
        raw_score = (
            (sum(news_scores)   / len(news_scores))   * weights["news"] +
            (sum(review_scores) / len(review_scores)) * weights["review"]
        )
    elif news_scores:
        raw_score = sum(news_scores) / len(news_scores)
    else:
        raw_score = sum(review_scores) / len(review_scores)

    # Incremental scoring: blend with previous run (EMA, 70% new / 30% historical)
    incremental = prev_score is not None
    if incremental:
        overall_score = 0.7 * raw_score + 0.3 * prev_score
    else:
        overall_score = raw_score

    keyword_split = extract_keyword_split(
        scored, top_n=5,
        business_name=business_name, location=location, category=category,
    )
    keywords = keyword_split["positive"][:3] + keyword_split["negative"][:3]
    if not keywords:
        keywords = extract_keywords(data, business_name=business_name, location=location, category=category)

    _update_learned_lexicon(keywords, data)

    summary     = extractive_summary(data, n_sentences=3)
    explanation = _explain_score(results, overall_score, keyword_split)

    return {
        "business_name":     business_name,
        "location":          location,
        "category":          category,
        "overall_sentiment": combined_label(overall_score),
        "overall_rating":    combined_rating(overall_score),
        "overall_score":     round(overall_score, 3),
        "items_analysed":    len(results),
        "keywords":          keywords,
        "keyword_split":     keyword_split,
        "summary":           summary,
        "explanation":       explanation,
        "incremental":       incremental,
        "breakdown":         results,
    }
