import re

import numpy as np
import pandas as pd
import transformers
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

data = pd.read_csv('book_reviews_sample.csv')

data['review_text_lowered'] = data['reviewText'].str.lower()
data['review_text_no_punct'] = data['review_text_lowered'].apply(lambda x: re.sub(r"([^\w\s])", "", x))

vader_sentiment = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data['review_text_no_punct'].apply(lambda x: vader_sentiment.polarity_scores(x)['compound'])
print(data.head())
bins = [-1,-0.1,0.1,1]
names = ['negative', 'neutral','positive']
data['vader_sentiment_label'] = pd.cut(data['vader_sentiment_score'],bins,labels = names)
transformer_pipeline = pipeline('sentiment-analysis')
transformer_labels = []
for review in data['review_text_no_punct'].values:
    sentiment_list = transformer_pipeline(review)
    sentiment_label = [sent['label'] for sent in sentiment_list]
    transformer_labels.append(sentiment_label)

data['transformer_sentiment'] = transformer_labels
print(data.head())