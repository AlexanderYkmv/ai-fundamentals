from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentence_1 = "i had a great time at the movie it was really funny"
sentence_2 = "i had a great time at the movie but the parking was terrible"
sentence_3 = "i had a great time at the movie but the parking wasn't great"
sentence_4 = "i went to see a movie"

s1_score = TextBlob(sentence_1)
s2_score = TextBlob(sentence_2)
s3_score = TextBlob(sentence_3)
s4_score = TextBlob(sentence_4)
print(s1_score.sentiment.polarity)
print(s2_score.sentiment.polarity)
print(s3_score.sentiment.polarity)
print(s4_score.sentiment.polarity)

vader_sentiment = SentimentIntensityAnalyzer()
print(sentence_1)
print(vader_sentiment.polarity_scores(sentence_1))