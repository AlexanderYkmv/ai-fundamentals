import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('wordnet')
data = pd.read_csv('tripadvisor_hotel_reviews.csv')
data['review_lower_case'] = data["Review"].str.lower()
print(data.head())
eng_stopwords = stopwords.words('english')
eng_stopwords.remove('not')
data['review_no_stop_words'] = data['review_lower_case'].apply(lambda x: ' '.join([word for word in x.split() if word not in (eng_stopwords)]))

#switch * to 'star'
data['review_no_stop_words_no_punct'] = data.apply(lambda x : re.sub(r"[*]", "star", x['review_no_stop_words']), axis=1)
data['review_no_stop_words_no_punct'] = data.apply(lambda x: re.sub(r"([^\w\s])", "", x['review_no_stop_words_no_punct']), axis=1)

data['tokenized'] = data.apply(lambda x: word_tokenize(x['review_no_stop_words_no_punct']), axis=1)
ps = PorterStemmer()
data['stemmed'] = data['tokenized'].apply(lambda tokens: [ps.stem(token) for token in tokens])
lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['tokenized'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])
tokens_clean = sum(data['lemmatized'], [])
unigrams = pd.Series(nltk.ngrams(tokens_clean,1)).value_counts()
print(unigrams)
bigrams = pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()
print(bigrams)