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

# Default Model is v0.1 
# version_number = 0.1
# NER_MODEL_PATH = BASE_PATH + f'/models/NER/{version_number}/'

# print(BASE_PATH)
# print(NER_MODEL_PATH)



# Load the NER Model
def load_model(version_number, path):
	
	model_name = f'ner_model_v{version_number}.sav'

	with open(path + model_name, 'rb') as file:
		ner_model = pickle.load(file)
	
	return ner_model

def load_ner_per_topic_dict(version_number, path):

	with open(path + f'topic_to_top5_ner_entities_v{version_number}.pkl', 'rb') as file:
		ner_topic_dict = pickle.load(file)

	return ner_topic_dict

# Predict NER Tags
def predict_ner_tags(input_text, nlp_corpus, ner_model):

	# Prepare raw input text into SentenceGetter parsable DataFrame
	x = text_cleaning.prep_query_for_NER(input_text)

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
	html_string = clean_displacy_html(html_string)

	# .replace(" ,"    ,   ",").replace("n ' t","n't").replace(" ."    ,   ".").replace("you ' re","you're")

	return html_string

def clean_displacy_html(html_string):

	pairs = [(" ,"   ,  ","),("dn t ","dn't "),(" ."   ,  "."),("an t ","an't "),("ou re ","ou're "),("I m ","I'm "),("hey re ","hey're "),("we re ","we're "),("We re","We're"),(" ll ","'ll "),("e s ","e's ")]

	for pair in pairs:
		html_string = html_string.replace(pair[0],pair[1])

	return html_string



