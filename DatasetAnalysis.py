from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import sklearn.model_selection as sk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
import math

pd.options.mode.chained_assignment = None



#print(df.nunique()) # get number of distinct values of each column
# 30971 unique users
#print(df.action.value_counts())
#print(df.action.count()) # 290258 actions taken

#df_searchValues['length_of_search'] = df_searchValues["searchValue"].apply(lambda x: ([len(w) for w in x.split()]))

#count = df_searchValues['searchValue'].str.split().str.len()

# preprocessing

df = pd.read_csv("Search Logs.csv")
df_events = pd.read_csv("UI events.csv")

df_searchValues = df[df.searchValue.notna()] # create dataframe of just entries with valid search values

x_raw = df_searchValues['searchValue'].values
y = df_searchValues['downloadType'].values

x_raw = [x.lower() for x in x_raw]

x_raw = [word_tokenize(x) for x in x_raw]

# build vocabulary and keeps the 2000 most frequent words
vocabulary = dict()
fdist = nltk.FreqDist()

for sentence in x_raw:
	for word in sentence:
		fdist[word] += 1

common_words = fdist.most_common(2000)

for idx, word in enumerate(common_words):
	vocabulary[word[0]] = (idx+1)

# convert each token in the dictionary into an index based representation
x_tokenized = list()

for sentence in x_raw:
	temp_sentence = list()
	for word in sentence:
		if word in vocabulary.keys():
			temp_sentence.append(vocabulary[word])
	x_tokenized.append(temp_sentence)

# pad sentences that do not fulfill the required length
pad_idx = 0
x_padded = list()
for sentence in x_tokenized:
	while len(sentence) < 20:
		sentence.insert(len(sentence), pad_idx)
	x_padded.append(sentence)

x_padded = np.array(x_padded, dtype=object)

# split data into train and test
x_train, x_test, y_train, y_test = sk.train_test_split(x_padded, y, test_size=.20, random_state=42)

# build model
embedding_layer = nn.Embedding(2000+1, 64, 0)

# convolution layers
kernel_1 = 2
kernel_2 = 3
kernel_3 = 4
kernel_4 = 5
stride = 2

conv_1 = nn.Conv1d(20, 32, kernel_1, stride)
conv_2 = nn.Conv1d(20, 32, kernel_2, stride)
conv_3 = nn.Conv1d(20, 32, kernel_3, stride)
conv_4 = nn.Conv1d(20, 32, kernel_4, stride)

# max pooling layers
pool_1 = nn.MaxPool1d(kernel_1, stride)
pool_2 = nn.MaxPool1d(kernel_2, stride)
pool_3 = nn.MaxPool1d(kernel_3, stride)
pool_4 = nn.MaxPool1d(kernel_4, stride)


embedding_size = 64
out_conv_1 = ((embedding_size - 1 * (kernel_1 - 1) - 1) / stride) + 1
out_conv_1 = math.floor(out_conv_1)
out_pool_1 = ((out_conv_1 - 1 * (kernel_1 - 1) - 1) / stride) + 1
out_pool_1 = math.floor(out_pool_1)

# Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
out_conv_2 = ((embedding_size - 1 * (kernel_2 - 1) - 1) / stride) + 1
out_conv_2 = math.floor(out_conv_2)
out_pool_2 = ((out_conv_2 - 1 * (kernel_2 - 1) - 1) / stride) + 1
out_pool_2 = math.floor(out_pool_2)

# Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
out_conv_3 = ((embedding_size - 1 * (kernel_3 - 1) - 1) / stride) + 1
out_conv_3 = math.floor(out_conv_3)
out_pool_3 = ((out_conv_3 - 1 * (kernel_3 - 1) - 1) / stride) + 1
out_pool_3 = math.floor(out_pool_3)

# Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
out_conv_4 = ((embedding_size - 1 * (kernel_4 - 1) - 1) / stride) + 1
out_conv_4 = math.floor(out_conv_4)
out_pool_4 = ((out_conv_4 - 1 * (kernel_4 - 1) - 1) / stride) + 1
out_pool_4 = math.floor(out_pool_4)


fc = nn.Linear((out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * 32, 1)

x = embedding_layer(x_train)

# Convolution layer 1 is applied
x1 = conv_1(x)
x1 = torch.relu(x1)
x1 = pool_1(x1)

# Convolution layer 2 is applied
x2 = conv_2(x)
x2 = torch.relu((x2))
x2 = pool_2(x2)

# Convolution layer 3 is applied
x3 = conv_3(x)
x3 = torch.relu(x3)
x3 = pool_3(x3)

# Convolution layer 4 is applied
x4 = conv_4(x)
x4 = torch.relu(x4)
x4 = pool_4(x4)

# The output of each convolutional layer is concatenated into a unique vector
union = torch.cat((x1, x2, x3, x4), 2)
union = union.reshape(union.size(0), -1)

# The "flattened" vector is passed through a fully connected layer
out = fc(union)
# Dropout is applied		
out = nn.Dropout(.25)(out)
# Activation function is applied
out = torch.sigmoid(out)
out.squeeze()



