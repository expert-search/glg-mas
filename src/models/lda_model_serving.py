# Load Utilities for Model Prediction
import pickle
import os
import numpy as np

# Load LDA Library
from gensim.models import Word2Vec
import gensim.corpora as corpora
import gensim

# From SRC | Load Helper Functions
# from src.text_cleaning import func_lemmatize, clean_query


# Declare model path
BASE_PATH = os.getcwd()

# Set Default Version Number
# version_number = str(0.1)
# MODEL_PATH = BASE_PATH + f'/models/LDA/{version_number}/model/'
# EMBEDDING_PATH = BASE_PATH + f'/models/LDA/{version_number}/embedding/'
# TOPIC_MAPPING_PATH = BASE_PATH + f'/models/LDA/{version_number}/topic_mapping/'


# print(BASE_PATH)
# print(MODEL_PATH)
# print(EMBEDDING_PATH)
# print(TOPIC_MAPPING_PATH)

# Load the LDA Model
def load_model(version_number, path):

	model_name = 'lda_model_v' + str(version_number)
	
	lda_model = gensim.models.ldamodel.LdaModel.load(path + model_name)

	return lda_model


# Load the ID2WORD Embedding
def load_id2word_embedding(version_number, embedding_path):
	
	embedding_name = f'id2word_v{str(version_number)}.pkl'
	
	with open(embedding_path + embedding_name, 'rb') as file:
		id2word = pickle.load(file)

	return id2word


# Predict Topics
def topic_predict(tokenized_query, embedding, lda_model, topics_dict):  

	# Clean up the text into a corpus
	# tokenized_input = clean_query(query)
	
	# Mapped embedding from Cleaned corpus text in list form
	corpus = embedding.doc2bow(tokenized_query)
	
	np.random.seed(4)
	
	# Model Predicts on cleaned corpus
	output = list(lda_model[corpus])
	
	
	# Post-Process Output for Display
	ordered = sorted(output,key=lambda x:x[1],reverse=True)
	

	# DEBUGGING
	print(len(ordered), ordered)


	# Issue Here
	primary_topic = ordered[0][0]
	
	threshold = 0.5
	
	secondary_topics = [pair[0] for pair in ordered[1:] if pair[1] / ordered[0][1] > threshold]
	
	# Promote Secondary Topics in case the Primary Topic was NULL
	# Check LDA output and try to bump any non-empty-string 2ndary topics to 1ary if necessary
	primary_topic, secondary_topics = check_topics(topics_dict, primary_topic, secondary_topics)

	# Will return integer topics
	return primary_topic, secondary_topics



def check_topics(topics_dict, primary_topic, secondary_topics):
    while True:
        # If the primary topic is an empty string and there are secondary topics...
        if (len(topics_dict[primary_topic]) == 0) & (len(secondary_topics) !=0):
            # Iterate through secondary topics
            for i in range(len(secondary_topics)):
                # Find the first secondary topic whose name is not an empty string
                if len(topics_dict[secondary_topics[i]]) != 0:
                    # Set that topic to primary status
                    primary_topic = secondary_topics.pop(i)
                    break
            else:
                break
        else:
            break
    return primary_topic, secondary_topics


# Print Topics
# def return_topics(topics, topics_dict):
#     print(f'primary topic: {topics_dict[topics[0]]}')

#     if secondary_topics:
#         print('-' * 10, '\n', 'other topics:')
#         for topic in topics[1]:
#             print(topics_dict[topic])


def lookup_secondary_topics(secondary_topic_codes, topic_dict):
	if secondary_topic_codes:
				
		# Capture Lookup from topic_dict in a list
		secondary_topics = [topic_dict[topic_code] for topic_code in secondary_topic_codes]
	else:
		pass

	return secondary_topics


def get_latest_version_from_path(path):
	# Assumes the LDA directory will be passed
	versions = os.listdir(path)
	version_number = max(versions)

	return float(version_number)


def load_topics_dict(version_number, path):
	
	# Version Number >= 0.1 - load from pickle
	# Load Latest Version 
	if not version_number:
		latest_version = get_latest_version_from_path(LDA_PATH)

	# Load Topic Mapping Dictionary from the Version Number folder in /LDA/
	filename = f'topics_dict_v{str(version_number)}.pkl'
	with open(path + filename, 'rb') as file:
		topics_dict = pickle.load(file)
	
	return topics_dict