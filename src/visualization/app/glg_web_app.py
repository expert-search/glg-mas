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

# Import SpaCy for NER Tagging
import spacy

# Models for Prediction
import src.features.text_cleaning as text_cleaning
import src.models.lda_model_serving as lda_endpoint
import src.models.ner_model_serving as ner_endpoint

# ----------------------------------------------------------------#
# Call the supplemental loading of components for nltk setup in the background
# text_cleaning.setup_nltk_downloaded_components()
# Setup Objects for NLTK to Parse Below
# wnl, stop, pattern = text_cleaning.session_nltk_objects_init()
wnl = WordNetLemmatizer()
wnl.lemmatize('cats')
stop = set(stopwords.words('english'))
pattern = r'(\w+)'
nlp_corpus = spacy.load('en_core_web_sm')


# Declare model path
BASE_PATH = os.getcwd()
LDA_MODEL_PATH = BASE_PATH + '/models/lda_250k/'
EMBEDDING_PATH = BASE_PATH + '/models/lda_embedding/'
NER_MODEL_PATH = BASE_PATH + '/models/ner_crf/'


# ------------------ Load Model in Background ----------------------- #
topic_dict = lda_endpoint.load_topics_dict()
id2word = lda_endpoint.load_id2word_embedding(embedding_name = 'id2word.pkl', embedding_path = EMBEDDING_PATH)
lda_model = lda_endpoint.load_model(model_name = '250k_ldamodel_185', path = LDA_MODEL_PATH)
ner_model = ner_endpoint.load_model(model_name = 'ner_crfsuite', path = NER_MODEL_PATH)

# ------------------- Begin Bage Rendering -------------------------- #

# Set Title for Page 
st.title('Predicting Topics from Unstructured Text')
st.write('---')


st.sidebar.title('Tool Selector')
st.sidebar.write('---')
analysis_type = st.sidebar.selectbox('Select Tool: ', ['Topics', 'Entities', 'Topic Clustering through Time'])



# Create an input text box to capture text input
st.header('Input Query')
# st.write('---')
text_to_parse = st.text_area('Submit a query ~ 1-2 sentences in length.')

# Topic Modeling Analysis
if analysis_type == 'Topics':
		
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

				topic_codes_array = lda_endpoint.topic_predict(cleaned_text, 
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
				secondary_topics = lda_endpoint.lookup_secondary_topics(secondary_topic_codes, topic_dict)
			
				with st.expander('Secondary Topics:'):

					for topic in secondary_topics:
						st.write(f'**{topic}**')







elif analysis_type == 'Entities':

	# NER Section - Load Named Entities Here
	# with col_right:
	st.subheader('Entities Found:')
	with st.container():
		
		# Fetch Model Predictions via Model API Call
		with st.spinner('Fetching Entities.'):

			### --------------------------ADD TEMP NER SECTION HERE-----------------------------#

			# html_string = '<div class="entities" style="line-height: 2.5; direction: ltr">In \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    March\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">tim</span>\n</mark>\n the joint study reported that it was extremely unlikely that the virus had been released in a laboratory accident . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Dr Ben Embarek\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">per</span>\n</mark>\n revealed that this conclusion did not come from a balanced assessment of all the relevant evidence but from a steadfast refusal by the \n<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Chinese\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">gpe</span>\n</mark>\n members of the joint study to support anything stronger . </div>'

			# Test Case 1: Technology
			# html_string = '<div class="entities" style="line-height: 2.5; direction: ltr">SoftBank needs a Plan B . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    One\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">tim</span>\n</mark>\n year on , prospects for its planned 54bn sale of \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    UK\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n chip designer \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Arm\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">per</span>\n</mark>\n to \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Nvidia\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n are souring . Antitrust watchdogs are circling . The \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    European Commission\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n is set to launch a formal competition probe while the \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    UK s Competition and Markets Authority\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n has dismissed \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Nvidia\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n s efforts as insufficient . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    China\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n , where \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Arm\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n has its own problems with a rogue joint venture partner , is likely to join the chorus . The deal is far from dead . A handful of \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Arm\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n clients have rallied round \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Nvidia\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    South Korea\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n s \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Samsung\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n provides precedent for vertical integration without compromising access to third party clients . Contractual remedies may save the day . </div>'
			if text_to_parse:
				# Predict NER Tags and return the HTML String to render
				html_string = ner_endpoint.predict_ner_tags(text_to_parse, nlp_corpus, ner_model)

				# Render tagged sentence with NER tags to UI
				st.markdown(html_string, unsafe_allow_html=True)
			else:
				st.write('Awaiting input text entry.')
				
			# Test Case 2: Healthcare
			# html_string = '<div class="entities" style="line-height: 2.5; direction: ltr">In \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Palermo\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n , \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Sicily\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n s capital , 80 percent of the hospitalized \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Covid\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n patients are unvaccinated , and a vast majority of those in the I . \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    C\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n . \n<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    U\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">org</span>\n</mark>\n . have not received a vaccine , said \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Dr\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Renato Costa\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">per</span>\n</mark>\n , the city s Covid emergency commissioner . Similar rates are observed throughout the region . If we had a higher vaccination rate , said \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Dr\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n . \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Costa\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n , our hospitals would be emptier . Local doctors said the drop in vaccination rates during the month \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    of August\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">tim</span>\n</mark>\n was related to the \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    summer\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">tim</span>\n</mark>\n holidays , a time when it is more difficult to distribute shots to the region , which has among \n<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">\n    Italy\n    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">geo</span>\n</mark>\n s lowest income and education levels . </div>'
			
			




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
