# Load Utilities for Model Prediction
import pickle



# Declare model path
MODEL_PATH = BASE_PATH + '/nlp-glg-mas/models/'


# Load the LDA Model
def load_model(model_name, path = MODEL_PATH):
	
	# Load the serialized model
	lda_model = pickle.load(open(path + 'lda_model.pkl', 'rb'))
	
	return lda_model
