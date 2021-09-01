# Basic Data Handling
import pandas as pd
import numpy as np
import re

# Import NLTK for Text Preprocessing
from nltk.tokenize import regexp_tokenize
from nltk import pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
import time
import pickle


# Must ensure we call this first!
def setup_nltk_downloaded_components():

	try:
		download('stopwords')
		download('averaged_perceptron_tagger')
		download('wordnet')
	except:
		raise 'Error instantiating NLTK objects.'

	return 'Successfully installed nltk supplemental components.'

# Lemmatize Text
def func_lemmatize(words, wnl):
    lemmatized = []
    
    for word, tag in pos_tag(words):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a','r','n','v'] else None
        
        # Call to 'wnl' object
        lemma = wnl.lemmatize(word,wntag) if wntag else word
        
        lemmatized.append(lemma)
    return lemmatized


# Return Cleaned Words
def clean_query(query, pattern, stop, wnl):

    # Call to regex pattern
    tokenized = regexp_tokenize(query, pattern)
    indiv_words = [word for word in tokenized if word.isalpha()]
    lemmatized = func_lemmatize(indiv_words, wnl)
    
    # Call to 'stop' for stopword removal
    words = [word.lower() for word in lemmatized if word not in stop]

    return words


# Run Text Cleaning Routine

#### Q: Mukesh/Gaurav: THIS SHOULD BE RUN ON INSTANCE STARTUP?
# Instantiate Lemmatizer, Stopwords Object, and Regex Pattern
def session_nltk_objects_init():

	wnl = WordNetLemmatizer()
	stop = set(stopwords.words('english'))
	pattern = r'(\w+)'

	return wnl, stop, pattern


# NER Preprocessing Functions
class NERSentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(), 
                                                           s['POS'].values.tolist(), 
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def prep_query_for_NER(phrase):
    split_query = re.findall(r"[\w']+|[.,!?;]", phrase)
    
    pos_tags = pos_tag(split_query)
    
    df_query = pd.DataFrame({'Sentence #':['Sentence: 1'] * len(pos_tags),
                            'Word':[pair[0] for pair in pos_tags],
                            'POS':[pair[1] for pair in pos_tags],
                            'Tag':[None] * len(pos_tags)})
       
    return df_query


