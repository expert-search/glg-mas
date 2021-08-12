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
MODEL_PATH = BASE_PATH + '/models/lda_250k/'
EMBEDDING_PATH = BASE_PATH + '/models/lda_embedding/'

print(BASE_PATH)
print(MODEL_PATH)
print(EMBEDDING_PATH)




# Load the LDA Model
def load_model(model_name = '250k_ldamodel_185', path = MODEL_PATH):
	
	# Load the serialized model
	# lda_model = pickle.load(open(path + 'lda_model.pkl', 'rb'))
	# model_name = '250k_ldamodel_185'

	lda_model = gensim.models.ldamodel.LdaModel.load(path + model_name)

	return lda_model


# Load the ID2WORD Embedding
def load_id2word_embedding(embedding_name='id2word.pkl', embedding_path = EMBEDDING_PATH):

	file = open(embedding_path + embedding_name, 'rb')
	id2word = pickle.load(file)

	return id2word


# Predict Topics
def topic_predict(tokenized_query, embedding, lda_model):  

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
    
    # Will return integer topics
    return [primary_topic, secondary_topics]


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


def load_topics_dict():
	topics_dict = {0:'ind',1:'sweden money-laundering',2:'libya politics',3:'gossip/celebrity life',4:'international business/economics',
               5:'mideast oil/business',6:'us politics',7:'international energy/renewables',8:'vietnam',9:'activism/protests',
               10:'nutrition/health science',11:'international diplomacy/politics',12:'washington politics',13:'celebrity life',
               14:'ind',15:'immunology/bloodborne diseases',16:'german-language business',17:'islamic geopolitics',
               18:'international oil/politics',19:'baseball/football',20:'art and museums',21:'international trade',
               22:'travel and tourism',23:'defence/military',24:'movies/video clips',25:'defence/weapons tech',
               26:'ind',27:'bankruptcy/corporate debt/m&a',28:'fashion',29:'heavy metals, minerals and mining',
               30:'gossip/celebrity life',31:'intl govt/politics',32:'no topic',33:'urban/city life',34:'gossip/celebrity life',
               35:'washington politics',36:'words and wordplay',37:'tax and tax policy',38:'ind',39:'beauty pageant',
               40:'stock earnings reports',41:'us census/treasury',42:'french current affairs',43:'obama',44:'beauty & makeup',
               45:'weather and natural disasters',46:None,47:'ind',48:'us politics',49:'showbiz awards',50:'global macroeconomics',
               51:'charitable donations/campaign contributions',52:'cybersecurity/hacking',53:'ind',54:'combat sports',
               55:None,56:'star wars franchise',57:'ugandan shilling',58:'sexual harrassment/assault',59:'consumer electronics',
               60:'global banking',61:'global c-suite executive',62:'trump',63:'climate and severe weather',64:'festivals and special occasions',
               65:'legal analysis',66:'immigration, refugees and asylum',67:'tech gadgets',68:'olympics',69:'wildfires/mexico politics',
               70:'nissan',71:'horoscopes',72:'washington politics',73:'myanmar',74:None,75:'ind',76:'global banking',77:'tennis',
               78:'telecommunications',79:'supreme court',80:None,81:'education',82:'rifles (russian language)',83:None,84:'ind',
               85:'wellness, self-help, relationships',86:'ind',87:'us sports',88:'game of thrones',
               89:'streaming/media svcs',90:'celebrities and sports stars',91:None,92:'gaming and VR',
               93:'intl corporate earnings',94:'future-tech, AI and crypto',95:'conservative politics',
               96:'Brazil business & politics',97:'tech vc, m&a, ipos, restructuring',98:'ind',99:'crimes/arrests',
               100:'physical and mental health',101:'college admissions scandal',102:'coronavirus',
               103:'international markets/monetary matters',104:'racing',105:'employment',106:'church and religion',
               107:'mid-east - US relations',108:'guns/shootings',109:'elon musk',110:'regulation, laws, govt oversight',
               111:'ind',112:'healthcare coverage/insurance',113:None,114:'social media providers',115:'sports',
               116:'trump white house investigations',117:'industrial workers and unions',118:'washington politics',
               119:'wildlife, aquatics and marine life',120:'ind',121:None,122:'stocks',123:'south american business',
               124:'le culture francais',125:'south asian/African business/politics',126:'music',127:'russia politics',
               128:'apps and mobile technology',129:'ind',130:'us/intl monetary policy',131:'jokes, parodies, trivia',
               132:'north korea relations',133:'misc lists and trivia',134:'aviation and aerospace',135:'homes and real estate',
               136:'espanol',137:'israeli-palestinian conflict',138:None,139:'celebs/reality tv stars',140:'astrophysics/space tech/sci fi',
               141:None,142:'biotechnology & pharmaceuticals',143:'ind',144:'south africa',145:'climate & environmental issues',
               146:'movies, tv, showbiz',147:'mobile devices, laptops, smart home tech',148:'performing arts',149:'ind',150:'celebrity hair & beauty',
               151:'ind',152:'recreational & medical marijuana',153:'pets, animals',154:'cross-border trade',155:'reproduction and birth control',
               156:'e-cigarettes, vaping and tobacco',157:'apple',158:'stock market & big business',159:'addiction and public health',
               160:'agriculture & farming',161:'food and drink',162:'space & space travel',163:'international politics & finance',
               164:'ind',165:'trump allies',166:'ind',167:'books, writing and writers',168:'politics & polls',169:'US elections',      
               170:'UK/EU economics, politics',171:'healthcare, insurance & retirement benefits',172:'cuba',173:'snackfoods, desserts & beverages',
               174:'misc lists and trivia',175:'desus & mero',176:'automakers & ride-sharing',177:'icebergs & glaciers',
               178:'ice hockey',179:'global big business',180:'washington politics',181:'China/HK business',182:'LGBTQ matters',
               183:'royalty and monarchy', 184:'retail'}
	
	return topics_dict