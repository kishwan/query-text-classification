import nltk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

class preprocessing:
	
	def __init__(self, num_words, seq_len):
		self.data = 'data/Search Logs.csv'
		self.num_words = num_words
		self.seq_len = seq_len
		self.vocabulary = None
		self.x_tokenized = None
		self.x_padded = None
		self.x_raw = None
		self.y = None
		self.x_train = None
		self.x_test = None
		self.y_train = None
		self.y_test = None
		
	def load_data(self):
		df = pd.read_csv(self.data)
		df_searchValues = df[df.searchValue.notna()] # create dataframe of just entries with valid search values

		self.x_raw = df_searchValues['searchValue'].values # create list that just contains the search queries in the dataset
		self.y = df_searchValues['downloadType'].values # create list that just contains if the corresponding search query resulted in a download or not
		
	def clean_text(self):
		self.x_raw = [x.lower() for x in self.x_raw] # set all strings to lowercase
		
	def text_tokenization(self):
		self.x_raw = [word_tokenize(x) for x in self.x_raw] # break each query into individual words for better text understand when building the model
	   
	def build_vocabulary(self): # build dictionary of the most frequent words that appear in the queries
	   self.vocabulary = dict()
	   fdist = nltk.FreqDist()
	   
	   for sentence in self.x_raw:
	      for word in sentence:
	         fdist[word] += 1
	         
	   common_words = fdist.most_common(self.num_words)
	   
	   for idx, word in enumerate(common_words):
	      self.vocabulary[word[0]] = (idx+1)
	      
	def word_to_idx(self): # using the dictionary from build_vocabulary(), each token is turned into its index based represntation (from last step of build_vocabulary)
	   self.x_tokenized = list()
	   
	   for sentence in self.x_raw:
	      temp_sentence = list()
	      for word in sentence:
	         if word in self.vocabulary.keys():
	            temp_sentence.append(self.vocabulary[word])
	      self.x_tokenized.append(temp_sentence)
	      
	def padding_sentences(self): # pad sentences so that they are all the same length (necessary step in text classification)
	   pad_idx = 0
	   self.x_padded = list()
	   
	   for sentence in self.x_tokenized:
	      while len(sentence) < self.seq_len:
	         sentence.insert(len(sentence), pad_idx)
	      self.x_padded.append(sentence)
	   
	   self.x_padded = np.array(self.x_padded, dtype=object)
	   
	def split_data(self): # split data into 80% train 20% test
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y, test_size=0.20, random_state=42)