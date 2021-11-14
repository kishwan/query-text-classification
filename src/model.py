import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class model(nn.ModuleList):

	def __init__(self, params):
		super(model, self).__init__()

		self.in_channels = params.in_channels
		self.num_words = params.num_words
		self.embedding_size = params.embedding_size
		self.dropout = nn.Dropout(0.25)
		
		self.kernel_1 = 2 # will determine needed kernel sizes later...
		self.kernel_2 = 3
		self.kernel_3 = 4
		self.kernel_4 = 5
		
		self.out_channels = params.out_channels

		self.stride = params.stride
		
		# define the embedding layer
		self.embedding = nn.Embedding(self.num_words + 1, self.embedding_size, padding_idx=0)
		
		# define the convolutional layers
		self.conv_1 = nn.Conv1d(self.embedding_size, self.out_channels, self.kernel_1, self.stride)
		self.conv_2 = nn.Conv1d(self.embedding_size, self.out_channels, self.kernel_2, self.stride)
		self.conv_3 = nn.Conv1d(self.embedding_size,self.out_channels, self.kernel_3, self.stride)
		self.conv_4 = nn.Conv1d(self.embedding_size, self.out_channels, self.kernel_4, self.stride)
		
		# define the max pooling layers
		self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
		self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
		self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
		self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
		
		# combine into linear layer
		self.fc = nn.Linear(self.in_features_fc(), 1)

		
	def in_features_fc(self):
		out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_conv_1 = math.floor(out_conv_1)
		out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
		out_pool_1 = math.floor(out_pool_1)
		
		out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_conv_2 = math.floor(out_conv_2)
		out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
		out_pool_2 = math.floor(out_pool_2)
		
		out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_conv_3 = math.floor(out_conv_3)
		out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
		out_pool_3 = math.floor(out_pool_3)
		
		out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_conv_4 = math.floor(out_conv_4)
		out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
		out_pool_4 = math.floor(out_pool_4)
		
		return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * self.out_channels
		
	def forward(self, x):
		# pass the sequence of tokens through the embedding layer
		x = self.embedding(torch.stack(x))
		
		# Convolution layer 1 is applied
		x = torch.permute(x, (0, 2, 1))
		print(x.size())
		#x = torch.transpose(x, -3, 0)
		x1 = self.conv_1(x)
		x1 = torch.relu(x1)
		x1 = self.pool_1(x1)
		
		# Convolution layer 2 is applied
		x2 = self.conv_2(x)
		x2 = torch.relu(x2)
		x2 = self.pool_2(x2)
	
		# Convolution layer 3 is applied
		x3 = self.conv_3(x)
		x3 = torch.relu(x3)
		x3 = self.pool_3(x3)
		
		# Convolution layer 4 is applied
		x4 = self.conv_4(x)
		x4 = torch.relu(x4)
		x4 = self.pool_4(x4)
		
		union = torch.cat((x1, x2, x3, x4), 2)
		union = union.reshape(union.size(0), -1)

		out = self.fc(union)

		out = self.dropout(out)

		out = torch.sigmoid(out)
		
		return out.squeeze()