from bs4 import BeautifulSoup
import requests
import re
import spacy
from collections import Counter
import matplotlib.pyplot as plt

urls = [
    "https://www.bbc.com/news",
    "https://edition.cnn.com",
    "https://www.thehindu.com"
]

def scrapeSite(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    para = [p.get_text(strip=True) for p in soup.find_all("p")]
    print("Scraped paragraphs:", para[:5])  # print only first 5 for readability
    return " ".join(para)

# Combine scraped text
text = [scrapeSite(url) for url in urls]
corpus = " ".join(text)

# Clean text
def cleanText(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

cleanedText = cleanText(corpus)

# NLP
nlp = spacy.load("en_core_web_sm")
doc = nlp(cleanedText)

# POS counts
pos_counts = Counter([token.pos_ for token in doc])
print("POS Counts:", pos_counts)

# NER counts
ent_counts = Counter([ent.label_ for ent in doc.ents])
print("NER Counts:", ent_counts)

# Count nouns
nouns = Counter()
for token in doc:
    if token.pos_ == "NOUN":
        nouns[token.pos_] += 1

# Plot noun counts
plt.bar(nouns.keys(), nouns.values())
plt.title("Noun Counts")
plt.ylabel("Count")
plt.xlabel("POS Tag")
plt.show()
