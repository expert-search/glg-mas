# Basic Data Handling
import pandas as pd
import numpy as np

# Import NLTK for Text Preprocessing
from nltk.tokenize import regexp_tokenize
from nltk import pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# import time
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
def func_lemmatize(words):
    lemmatized = []
    
    for word, tag in pos_tag(words):
        wntag = tag[0].lower()
        wntag = wntag if wntag in ['a','r','n','v'] else None
        
        # Call to 'wnl' object
        lemma = wnl.lemmatize(word,wntag) if wntag else word
        
        lemmatized.append(lemma)
    return lemmatized


# Return Cleaned Words
def clean_query(query):

    # Call to regex pattern
    tokenized = regexp_tokenize(query, pattern)
    indiv_words = [word for word in tokenized if word.isalpha()]
    lemmatized = func_lemmatize(indiv_words)
    
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


