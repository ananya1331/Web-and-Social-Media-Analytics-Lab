import re
import nltk
nltk.download('punkt')

texts = [
    "I ❤️ Python! Check this out: https://python.org",
    "Data Science is 🔥🔥 #MachineLearning",
    "Follow me 👉 http://example.com 😎",
]

def clean_and_tokenize(text_list):
    clean_sentences = []
    for text in text_list:
        # 1. Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # 2. Remove emojis and special chars (keep only letters and spaces)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # 3. Lowercase
        text = text.lower()
        # 4. Tokenize
        tokens = nltk.word_tokenize(text)
        clean_sentences.append(tokens)
    return clean_sentences

tokens_list = clean_and_tokenize(texts)
print(tokens_list)
