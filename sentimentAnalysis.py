import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import matplotlib.pyplot as plt

# -------------------------
# 1. Download VADER Lexicon
# -------------------------
nltk.download("vader_lexicon")

# -------------------------
# 2. Sample Reviews
# -------------------------
reviews = [
    "I absolutely love this phone, the camera is amazing!",
    "The battery life is terrible and it keeps overheating.",
    "It's okay, does the job but nothing special.",
    "Excellent build quality and very fast performance.",
    "Worst purchase ever! Completely useless after a week.",
    "Pretty decent for the price, but could be better."
]

# ================================================
# PART A: VADER Sentiment Analysis
# ================================================
print("\n=== VADER Sentiment Analysis ===")
vader = SentimentIntensityAnalyzer()

vader_scores = []

for review in reviews:
    score = vader.polarity_scores(review)
    vader_scores.append(score["compound"])
    sentiment = "Positive" if score["compound"] > 0.05 else "Negative" if score["compound"] < -0.05 else "Neutral"
    print(f"Review: {review}")
    print(f"Score: {score} → {sentiment}")
    print("*" * 50)

# ================================================
# PART B: TextBlob Sentiment Analysis
# ================================================
print("\n=== TextBlob Sentiment Analysis ===")
textblob_scores = []

for review in reviews:
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    textblob_scores.append(polarity)
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    print(f"Review: {review}")
    print(f"Score: {polarity:.3f} → {sentiment}")
    print("*" * 50)

# ================================================
# PART C: spaCy + spaCyTextBlob Sentiment Analysis
# ================================================
print("\n=== spaCy Sentiment Analysis ===")

# ✅ Load and set up NLP pipeline
nlp = spacy.load("en_core_web_sm")
if "spacytextblob" in nlp.pipe_names:
    nlp.remove_pipe("spacytextblob")
nlp.add_pipe("spacytextblob")

spacy_scores = []

for review in reviews:
    doc = nlp(review)
    polarity = doc._.polarity
    spacy_scores.append(polarity)
    sentiment = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
    print(f"Review: {review}")
    print(f"Polarity: {polarity:.3f} → {sentiment}")
    print("-" * 50)

# ================================================
# PART D: Visualization with Matplotlib
# ================================================
labels = [f"R{i+1}" for i in range(len(reviews))]
x = range(len(reviews))

plt.figure(figsize=(10, 6))
plt.plot(x, vader_scores, marker='o', label="VADER Compound")
plt.plot(x, textblob_scores, marker='s', label="TextBlob Polarity")
plt.plot(x, spacy_scores, marker='^', label="spaCy Polarity")

plt.xticks(x, labels, rotation=30)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Review")
plt.ylabel("Sentiment Polarity")
plt.title("Sentiment Analysis Comparison")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
