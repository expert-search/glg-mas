import nltk

# Load NLTK Dependencies on Docker Container Initialization
def init_nltk_dependencies():
	nltk.download('stopwords')
	nltk.download('averaged_perceptron_tagger')
	nltk.download('wordnet')


if __name__ == '__main__':
	init_nltk_dependencies()
	print('Loaded NLTK Dependencies.')
