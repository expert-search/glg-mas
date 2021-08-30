# Load Data Handling Utitlies
import pandas as pd
import numpy as np
import pickle
import os

# Load Libraries for NER Prediction
from nltk import pos_tag
import spacy
from spacy.tokens import Doc
from spacy import displacy
import sklearn_crfsuite
import re

# Load Text Preprocessing Utilities
from src.features import text_cleaning

# Declare model path
BASE_PATH = os.getcwd()
NER_MODEL_PATH = BASE_PATH + '/models/ner_crf/'

print(BASE_PATH)
print(NER_MODEL_PATH)



# Load the NER Model
def load_model(model_name = 'ner_crfsuite', path = NER_MODEL_PATH):
	
	if model_name == 'ner_crfsuite':
		ner_model = pickle.load(open(path + '0.1-maf-crf_ner_model.sav', 'rb'))
	
	return ner_model


# Predict NER Tags
def predict_ner_tags(input_text, nlp_corpus, ner_model):

	# Prepare raw input text into SentenceGetter parsable DataFrame
	x = text_cleaning.prep_query_for_NER(input_text)

	print(type(x))

	# Create an instance of the SentenceGetter class from the raw input text
	getter_query = text_cleaning.NERSentenceGetter(x)
	
	# Parse the sentences to build the NER Input Features
	sentences_query = getter_query.sentences

	# Build the input features for the NER model
	X_query = [text_cleaning.sent2features(s) for s in sentences_query]
	
	# Splice the sentence into words to match with entities predicted from the NER model
	X_words = [s[0] for s in sentences_query[0]]

	# Calculate predictions for NER Tags using CRFSuite
	pred = ner_model.predict(X_query)

	# Parse the Predicted Entities into a list of tuples pairing the recognized entities with their original word
	ents = list(zip(pred[0], X_words))

	# Extract the tags for all entities, with the tags as well as the non-tags
	named_ent_tags = [tag.upper() for tag, word in ents]


	# Create a SpaCy Doc using the custom tags from CRFSuite and the original sequence of input words
	doc = Doc(nlp_corpus.vocab, words= X_words, ents = named_ent_tags)

	# Render the HTML string as output for the input text
	html_string = displacy.render(doc, style = "ent", jupyter = False)

	return html_string


