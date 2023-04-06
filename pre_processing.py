# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 08:45:19 2021

@author: Ali
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk 
import string
import re


df=pd.read_csv('C:/Users/Ali/.spyder-py3/againtwint/Selected and new features_decoded.csv').shape


def load_data():
     df=pd.read_csv('C:/Users/Ali/.spyder-py3/againtwint/Selected and new features_decoded.csv')
     return df

# # # #load some data
#tweets_of_uni = load_data()
tweet_df = load_data()


#print('Dataset size:',tweet_df.shape)
#print('Columns are:',tweet_df.columns)



#Information=tweet_df.info()

#Loading ids and tweets
df  = pd.DataFrame(tweet_df)


#Punctuation
string.punctuation

#Removing Punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

df['Tweet_punct'] = df['tweet'].apply(lambda x: remove_punct(str(x)))



#Tokenization of Data
def tokenization(text):
     text = re.split('\W+', text)
     return text

df['Tweet_tokenized'] = df['Tweet_punct'].apply(lambda x: tokenization(str(x).lower()))



#Stopwords removing
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
   text = [word for word in text if word not in stopword]
   return text
    
df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(str(x)))




#Stemming and Lammitization


#Stemming
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Tweet_stemmed'] = df['tweet'].apply(lambda x: stemming(str(x)))



#Lemmatization
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Tweet_lemmatized'] = df['tweet'].apply(lambda x: lemmatizer(str(x)))




def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text
df['Tweet_cleaned'] = df['tweet'].apply(lambda x: clean_text(str(x)))



df.drop_duplicates(subset='tweet',inplace=True)

df.info()

df.to_csv('C:/Users/Ali/.spyder-py3/againtwint/Selected and new features_decoded_pre_processed.csv')

#['id', 'tweet', 'Tweet_punct','Tweet_tokenized', 'Tweet_nonstop', 'Tweet_stemmed','Tweet_lemmatized' , 'Tweet_cleaned']

    

# from nltk.probability import FreqDist
# fdist = FreqDist(tokenized_word)
# print(fdist)
# <FreqDist with 25 samples and 30 outcomes>
# fdist.most_common(2)




#nltk.pos_tag(tokens)
