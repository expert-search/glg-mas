import streamlit as st

# Data Libraries
import pickle
import numpy as np
import pandas as pd

# Standard Python Libraries
import os
import time

# Import NLTK for Text Preprocessing
from nltk.tokenize import regexp_tokenize
from nltk import pos_tag, download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Models for Prediction
import src.features.text_cleaning as text_cleaning
import src.models.lda_model_serving as model_endpoint

# ----------------------------------------------------------------#
# Call the supplemental loading of components for nltk setup in the background
# text_cleaning.setup_nltk_downloaded_components()
# Setup Objects for NLTK to Parse Below
# wnl, stop, pattern = text_cleaning.session_nltk_objects_init()
wnl = WordNetLemmatizer()
wnl.lemmatize('cats')
stop = set(stopwords.words('english'))
pattern = r'(\w+)'


# Declare model path
BASE_PATH = os.getcwd()
MODEL_PATH = BASE_PATH + '/models/lda_250k/'
EMBEDDING_PATH = BASE_PATH + '/models/lda_embedding/'


# ------------------ Load Model in Background ----------------------- #
topic_dict = model_endpoint.load_topics_dict()
id2word = model_endpoint.load_id2word_embedding(embedding_name = 'id2word.pkl', embedding_path = EMBEDDING_PATH)
lda_model = model_endpoint.load_model('250k_ldamodel_185', path = MODEL_PATH)


# ------------------- Begin Bage Rendering -------------------------- #

# Set Title for Page 
st.title('Predicting Topics from Unstructured Text')
st.write('---')


st.sidebar.title('Tool Selector')
st.sidebar.write('---')
analysis_type = st.sidebar.selectbox('Select Tool: ', ['Topics', 'Entities', 'Topic Clustering through Time'])

# Topic Modeling Analysis
if analysis_type == 'Topics':
	
	# Create an input text box to capture text input
	st.header('Input Query')
	# st.write('---')
	text_to_parse = st.text_area('Which topics lay embedded within? Submit a query ~ 1-2 sentences in length.')
	
	# if st.button('Process Text'):
	# 	st.write('Now Processing...')

	# Double check if this has captured anything and will delay page load below?
	# if text_to_parse:
	# 	st.write(text_to_parse)
				
	# Clean the lemmatized text for dispatch to model
	cleaned_text = text_cleaning.clean_query(text_to_parse, pattern, stop, wnl)

	# Print to check
	# st.write(cleaned_text)

	# Set up Columns to Print Output
	# col_left, col_right = st.columns(2)


	# Topic Modeling Section
	# with col_left:
	st.header('Topics Found:')
	st.write('---')
	with st.container():
		
		# Fetch Model Predictions via Model API Call
		with st.spinner('Fetching Topics.'):

			# Delay to simulate model API Call
			# time.sleep(3)

			# Predict Topics - only if text has been entered
			if len(cleaned_text) > 0:

				topic_codes_array = model_endpoint.topic_predict(cleaned_text, 
																embedding = id2word, 
																lda_model = lda_model)

				# DEBUG to Console
				# print('\n')
				# print(topic_codes_array)
				# print('\n')

				# Returns a string
				# st.write(topic_codes_array)

				primary_topic = topic_dict[topic_codes_array[0]]
				
				# Create an output text box with a list of topics returned from the model
				# Ensure topic_array contains actual tokenized text - (length >= 1)
				if len(topic_codes_array) >= 1:
					
					st.write(f'Primary Topic: **{primary_topic}**')

		# Expand Details if necessary
		if len(cleaned_text) > 0:

			# Secondary Parsing - Collect Codes as a list
			secondary_topic_codes = topic_codes_array[1]

			if len(secondary_topic_codes) > 0:
				secondary_topics = model_endpoint.lookup_secondary_topics(secondary_topic_codes, topic_dict)
			
				with st.expander('Secondary Topics:'):

					for topic in secondary_topics:
						st.write(f'**{topic}**')



elif analysis_type == 'Entities':

	# NER Section - Load Named Entities Here
	# with col_right:
	st.subheader('Entities Found:')
	st.write('---')
	with st.container():
		
		# Fetch Model Predictions via Model API Call
		with st.spinner('Fetching Entities.'):

			# time.sleep(3)
			st.write('1. DISPLAY SPACY ENTITY TAGGED OUTPUT HERE')


			st.write('2. Output List of Named Entities here.')




elif analysis_type == 'Topic Clustering through Time':
	# Dendrogram Date Range Selection Section
	st.header('Hierarchical Dendrogram - Version 2')
	st.write('---')
	st.write('Starting year must come before ending year')
	start_year = st.slider('Select a Starting Year', min_value=2004, max_value=2021)
	end_year = st.slider('Select an Ending Year', min_value=2004, max_value=2021)

	st.write(f'Analyzing topic clusters for the period from {start_year} to {end_year}')


# ----------------- Footer -------------------- #
st.write('---')
st.markdown('**Team GLG**: Mark + Spencer')
