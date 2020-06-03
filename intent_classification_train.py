								##############  Training Code ################

import pandas as pd
import numpy
import h5py as h5
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet, stopwords
from sklearn import metrics
from nltk.stem import WordNetLemmatizer
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.layers import Flatten
from keras.layers.core import Dense, Dropout,Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# from sklearn.cross_validation import train_test_split
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot
from keras. callbacks import TensorBoard
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import time

def clean_data(text):
	wordnet_lemmatizer = WordNetLemmatizer()
	stops = stopwords.words('english')
	nonan = re.compile(r'[^a-zA-Z ]')
	output = []
	for i in range(len(text)):
#		print("original",text[i])
		sentence = nonan.sub('', text[i])
		words = word_tokenize(sentence.lower())
		filtered_words = [w for w in words if not w.isdigit() and not w in stops and not w in string.punctuation]
		tags = pos_tag(filtered_words)
		cleaned = ''
#		print("==>",tags)
		pos = ['NN','NNS','NNP','NNPS','RP','MD','FW','VBZ','VBD','VBG','VBN','VBP','RBR','JJ','RB','RBS','PDT','JJ','JJR','JJS','TO','VB']
		nos = ['NN','NNS','NNP','NNPS']
		for word, tag in tags:
			if tag in pos:
				cleaned = cleaned + wordnet_lemmatizer.lemmatize(word) + ' '
		print(cleaned)
#		time.sleep(5)
		output.append(cleaned.strip())
	return output

def removePunc(sentence):
	remove_punc = re.sub(r'[^\w\s]','',sentence)
	#return corpus with out punctuation.
	return remove_punc

def stemmed(Text):
	ps = PorterStemmer()
	cleanText = []
	for word in Text:
		cleanText.append(ps.stem(word))
	return cleanText

def prepareX(corpus):

	print(type(corpus))
	new_corpus = []
	remove_words = ['iPhone', 'iphone', '8', 'x', 'iOS', '11', 'oppo', 'Note',
					'note', 'Apple',
					'apple']
	stop_words = set(stopwords.words('english'))

	for i in remove_words:
		stop_words.add(i)
	for sentence in corpus:
		try:
			sentence = removePunc(sentence)
			# print("=>",sentence)

			sentence = word_tokenize(sentence)

			for w in sentence:

				if w in stop_words:
					sentence.remove(w)

			sentence = ' '.join(sentence)
			new_corpus.append(sentence)

		except:
			# print("exception fired")
			new_corpus.append(sentence)
			# continue

	return new_corpus



def prepareY(dfy):
		dfy = dfy.str.lower()
		dfy[dfy == 'pi'] = 'yes'
		dfy[dfy == 'undefined'] = 1
		dfy[dfy == 'yes'] = 2
		dfy[dfy == 'no'] = 0
		print(dfy)
		print(len(dfy))

		return dfy




def train(x,y):

	print(x)
	tfidf_vectorize = TfidfVectorizer()
	vectors = tfidf_vectorize.fit_transform(x)
	features = tfidf_vectorize.get_feature_names()

	dense = vectors.todense()
	denselist = dense.tolist()
	x = pd.DataFrame(denselist, columns=features)



	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)



	#	Fitting XGBoost to the Training set
	from xgboost import XGBClassifier

	# Loading pre-trained model
	model = load_model('model14.model')

	classifier = XGBClassifier()
	classifier.fit(x_train, y_train)

	# Predicting the Test set results
	y_pred = classifier.predict(x_test)

	 # Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
    
    # save and print accuracy
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print(accuracy) 
	# accuracies.mean()
	# accuracies.std()
	# print(accuracies)
	# classifier.save_model('model.h5')


def nn_lstm(x,y):
	# lstx = list(x)
	tk = Tokenizer(nb_words=40000, lower=True, split=" ")
	tk.fit_on_texts(x)
	print(tk)
	# testy = testy.iloc[:].values
	############################################################################################################################################

	trainx = tk.texts_to_sequences(x)
	print(trainx)
	############################################################################################################################################

	max_len = 47
	print(x)
	x = pad_sequences(trainx)
	max_features = 40000
	# y = list(y)
	from sklearn.model_selection import train_test_split
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
	print('Build model...')
	model = Sequential()
	model.add(Embedding(max_features, 128, input_length=max_len))
	# model.add(GaussianNoise(0.2))
	model.add(LSTM(256,return_sequences=True))
	model.add(LSTM(256,return_sequences=True))
	model.add(LSTM(256,return_sequences=True))
	model.add(LSTM(128,return_sequences=True))
	model.add(LSTM(128,return_sequences=True))
	model.add(LSTM(128,return_sequences=True))
	model.add(LSTM(56))
	model
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.summary()
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
	# plot(model, to_file='model.png', show_shapes=True)
	tb_cb = TensorBoard(log_dir='./Graph', histogram_freq=0,
						write_graph=True, write_images=True)
	model_history = model.fit(x_train, y=y_train, batch_size=128, nb_epoch=3, verbose=1, validation_data=(x_test,y_test),
							  callbacks=[tb_cb])
	############################################################################################################################################
	model.save('temporary_model_23417.h5')
	model = load_model('/home/shivansh/Desktop/FYP-II/temporary_model_23417.h5')

	y_pred = model.predict(x_test)
	print(y_pred)
	from sklearn.metrics import confusion_matrix,classification_report
	# cm = confusion_matrix(y_test, y_pred)
	# print(cm)
	cr = classification_report(y_test, y_pred)
	print(cr)


print(pos_tag("iphone x"))
########################################################## MAIN ###########################################

df = pd.read_csv("/home/shivansh/Desktop/FYP-II/data/combined_csv.csv")
print(len(df))
df.drop_duplicates(inplace=True)
print(len(df))

df.columns = ["index","class","tweets",""]
print(df)

df['class'] = prepareY(df['class']).copy()
#df.drop([1652], inplace=True)
x = df.iloc[:,2].values
y = df.iloc[:,1].values
y=y.astype('int')
y_view = list(y)
x = clean_data(x)



train(x,y)
