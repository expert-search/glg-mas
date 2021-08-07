import streamlit as st

# Data Libraries
import pickle
import numpy as np
import pandas as pd

# Standard Python Libraries
import os
import time



# ----------------------------------------------------------------#



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


	# Set up Columns to Print Output
	col_left, col_right = st.columns(2)


	# Topic Modeling Section
	with col_left:
		st.subheader('Topics Found:')
		st.write('---')
		with st.container():
			
			# Fetch Model Predictions via Model API Call
			with st.spinner('Fetching Topics.'):

				# Delay to simulate model API Call
				# time.sleep(3)

				# Create an output text box with a list of topics returned from the model
				st.write(text_to_parse)

			# Expand Details if necessary
			with st.expander('Additional Details:'):
				st.write('Further details related to the LDA model output will be shown here')


	# NER Section - Load Named Entities Here
	with col_right:
		st.subheader('Entities Found:')
		st.write('---')
		with st.container():
			
			# Fetch Model Predictions via Model API Call
			with st.spinner('Fetching Entities.'):

				# time.sleep(3)
				st.write(text_to_parse)



elif analysis_type == 'Topic Clustering through Time':
	# Dendrogram Date Range Selection Section
	st.header('Hierarchical Dendrogram')
	st.write('---')
	st.write('Starting year must come before ending year')
	start_year = st.slider('Select a Starting Year', min_value=2004, max_value=2021)
	end_year = st.slider('Select an Ending Year', min_value=2004, max_value=2021)

	st.write(f'Analyzing topic clusters for the period from {start_year} to {end_year}')

	''' _This_ is some __Markdown__ '''


# ----------------- Footer -------------------- #
st.write('---')
st.markdown('**Team GLG**')
st.write('Mark + Spencer')
