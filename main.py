from transformers import pipeline

# Load FinBERT model on startup
print("Loading models...")
sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")
print("Models ready.\n")


def get_rating(label, score):
    """Convert FinBERT sentiment and confidence score to a 1–5 numeric rating."""
    if label == "negative":
        return 1 if score >= 0.75 else 2
    if label == "neutral":
        return 3
    # positive
    return 5 if score >= 0.75 else 4


def analyse(text):
    """Run FinBERT sentiment analysis and return label, confidence, and rating."""
    result = sentiment_model(text)[0]
    label = result["label"]   # "positive", "negative", or "neutral"
    score = result["score"]
    rating = get_rating(label, score)
    return label, score, rating


def main():
    print("=== FinBERT Financial Sentiment Analyser ===")
    print("Paste a financial news sentence or review, then press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Input: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        print("\nAnalysing...")
        label, score, rating = analyse(text)

        print(f"\n  Sentiment : {label.capitalize()} ({score:.0%} confidence)")
        print(f"  Rating    : {rating} / 5")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
