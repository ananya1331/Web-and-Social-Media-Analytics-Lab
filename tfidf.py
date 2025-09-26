# Goal: Clean text (Tokenize + Regex), n-grams, lexical diversity, tf idf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.util import ngrams
import pandas as pd 
from collections import Counter 

nltk.download('punkt')
nltk.download('stopwords')

vectorizer = TfidfVectorizer()

docs = [
    "Data Science is fun and essential for modern careers.",
    "Machine learning and AI are overlapping fields.",
    "Python is popular, versatile, and easy!"
]

# Data Cleaning
def cleaning(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = re.sub['[^\w\s+]','',for t in tokens]
    tokens = [t for t in tokens in t not in stopwords('english')]
    return tokens

# N-Grams
for i, doc in enumerate(docs):
    print(f"N-grams for document{i}:")
    for n in [1,2,3]:
        ng = list(ngrams(tokens,n))
        print(f"{n}-grams:{ng}")

# TF IDF
x = vectorizer.fit_transform(docs)
df = pd.DataFrame(x.toarray(), columns = vectorizer.get_feature_names_out())
print(df)



