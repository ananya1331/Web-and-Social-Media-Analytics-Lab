from nltk.stem import PorterStemmer, SnowballStemmer
from collections import Counter
import snscrape.modules.twitter as sntwitter

query = "Data Science"

tweets = [tweet.content for tweet in sntwitter.TwitterSearchScraper(query).get_items()][:10]

porter = PorterStemmer()
snowball = SnowballStemmer()

porterStemming = [porter.stem(w) for tweet in tweets for w in nltk.word_tokenize(tweet.lower())]
snowballStemming = [snowball.stem(w) for tweet in tweets for w in nltk.word_tokenize(tweet.lower())]

freq_porter = Counter(porterStemming)
freq_snowball = Counter(snowballStemming)

print("\nTop 10:", freq_porter.most_common(10))
print("\nTop 10:", freq_snowball.most_common(10))