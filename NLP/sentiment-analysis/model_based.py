import transformers
from transformers import pipeline

sentence_1 = "i had a great time at the movie it was really funny"

sentiment_pipeline = pipeline("sentiment-analysis")
print(sentiment_pipeline(sentence_1))