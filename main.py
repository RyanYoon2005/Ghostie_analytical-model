from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load FinBERT and summarization model
sum_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")

def summarise(text):
    """Summarise text using DistilBART."""
    inputs = sum_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    ids = sum_model.generate(inputs["input_ids"], max_length=80, min_length=20, do_sample=False)
    return sum_tokenizer.decode(ids[0], skip_special_tokens=True)

def get_rating(label, score):
    """Convert FinBERT sentiment and confidence score to a 1-5 numeric rating."""
    if label == "negative":
        return 1
    if label == "neutral":
        return 3
    if label == "positive":
        return 5

def analyse(text):
    """Summarise the text, run FinBERT, and return results."""
    word_count = len(text.split())
    summary = summarise(text) if word_count >= 30 else text

    result = sentiment_model(text)[0]
    print(result)
    label = result["label"]
    score = result["score"]
    rating = get_rating(label, score)

    return summary, label, score, rating

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
        summary, label, score, rating = analyse(text)

        print(f"\n  Summary   : {summary}")
        print(f"  Sentiment : {label.capitalize()} ({score:.0%} confidence)")
        print(f"  Rating    : {rating} / 5")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
