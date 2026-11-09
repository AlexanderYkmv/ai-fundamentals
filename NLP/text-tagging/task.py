import re

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

eng_stopwords = stopwords.words('english')
data = pd.read_csv('bbc_news.csv')
data_titles = pd.DataFrame(data["title"])

data_titles['lowered_titles'] = data_titles['title'].str.lower()

data_titles['no_stopwords'] = data_titles['lowered_titles'].apply(lambda x:' '.join([word for word in x.split() if word not in (eng_stopwords)]))

data_titles['no_stopwords_no_punct'] = data_titles['no_stopwords'].apply(lambda x: re.sub(r"([^\w\s])", "", x))

data_titles['tokens_raw'] = data_titles['title'].apply(lambda x: word_tokenize(x))
data_titles['tokens_clean'] = data_titles['no_stopwords_no_punct'].apply(lambda x: word_tokenize(x))

lemmatizer = WordNetLemmatizer()
data_titles['tokens_clean_lemmatized'] = data_titles['tokens_clean'].apply(lambda x:[lemmatizer.lemmatize(token) for token in x])
print(data_titles.head())

tokens_raw_list = sum(data_titles['tokens_raw'], [])
tokens_clean_list = sum(data_titles['tokens_clean_lemmatized'], [])

nlp = spacy.load('en_core_web_sm')
spacy_doc = nlp(' '.join(tokens_raw_list))
pos_df = pd.DataFrame(columns=['token','pos_tag'])
for token in spacy_doc:
    pos_df = pd.concat([pos_df, pd.DataFrame.from_records([{'token': token.text,'pos_tag':token.pos_}])], ignore_index=True)
    
pos_df_counts = pos_df.groupby(['token','pos_tag']).size().reset_index(name='counts').sort_values(by='counts',ascending=False)

nouns = pos_df_counts[pos_df_counts.pos_tag == "NOUN"][0:10]

ner_df = pd.DataFrame(columns=['token','ner_tag'])
for token in spacy_doc.ents:
    if pd.isna(token.label_) is False:
        ner_df = pd.concat([ner_df, pd.DataFrame.from_records([{'token': token.text, 'ner_tag': token.label_}])], ignore_index=True)