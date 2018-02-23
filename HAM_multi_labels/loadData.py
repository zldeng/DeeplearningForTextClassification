#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-02-08 10:33:41
'''
import sys
import gensim
import numpy as np


def batch_iter(data, batch_size, shuffle=True):
	"""
	Generates a batch iterator for a dataset.
	"""
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	#for epoch in range(num_epochs):
	# Shuffle the data at each epoch
	if shuffle:
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_data = data[shuffle_indices]
	else:
		shuffled_data = data
	
	for batch_num in range(num_batches_per_epoch):
		start_index = batch_num * batch_size
		end_index = min((batch_num + 1) * batch_size, data_size)
		yield shuffled_data[start_index:end_index]

def loadVocabulary(vocab_file):
	word2idx = {}
	idx2word = {}

	with open(vocab_file) as f:
		for line in f:
			line_list = line.strip().split('\t')

			if len(line_list) != 2:
				print('ErrorLine:',line)
				continue

			word = line_list[0]
			word_idx = int(line_list[1])

			word2idx[word] = word_idx
			idx2word[word_idx] = word

	print ('Load vocabulary done. data_cnt:',len(word2idx))

	return word2idx,idx2word

def generateEmbeddingMatrixFromWord2vec(word2vec_model_file,vocab_file):
	word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_file,
			binary = False,unicode_errors = 'ignore')

	word2idx,idx2word = loadVocabulary(vocab_file)

	vocab_size = len(word2idx)
	embedding_size = word2vec_model.vector_size

	embedding_matrix = np.zeros((vocab_size,embedding_size))

	find_cnt = 0
	for word,idx in word2idx.items():
		if idx >= vocab_size:
			continue

		if word in word2vec_model:
			find_cnt += 1
			embedding_matrix[idx] = word2vec_model[word]

	print('word in embedding matrix:',find_cnt)

	return embedding_matrix


def testEmbedding():
	word2vec_model_file = './data/henan_ham/w2v_model_STD_NOSEG.vec'
	voc_file = './data/henan_ham/voc.txt'

	embedding_matrix = generateEmbeddingMatrixFromWord2vec(word2vec_model_file,voc_file)

	print(embedding_matrix.shape)



def loadDataFromFile(doc_file,vocab_file,
		sentence_num_per_doc = 30,
		word_num_per_sentence = 80,
		label_num = 42):
	
	word2id,id2word = loadVocabulary(vocab_file)

	doc_data = []
	doc_label = []

	data_X = []
	data_Y = []
	with open(doc_file) as f:
		for line in f:
			line_list = line.strip().split('\t')

			if len(line_list) != 2:
				print('Error line:',line)
				continue

			sentence_list = line_list[0].strip().split(' ')
			label_list = [int(label_idx) for label_idx in line_list[1].split(',')]
			

			sample_x = np.zeros((sentence_num_per_doc,word_num_per_sentence))
			for sentence_idx in range(min(len(sentence_list),sentence_num_per_doc)):
				sentence = sentence_list[sentence_idx]
				for wd_idx in range(min(len(sentence),word_num_per_sentence)):
					cand_word = sentence[wd_idx]

					word_idx_in_voc = word2id.get(cand_word,0)
					sample_x[sentence_idx][wd_idx] = word_idx_in_voc

			sample_y = np.zeros(label_num)
			sample_y[label_list] = 1
			
			data_X.append(sample_x)
			data_Y.append(sample_y)
	
	data_X = np.array(data_X)
	data_Y = np.array(data_Y)

	return data_X,data_Y

def testloadDataFromFile():
	doc_file = './data/henan_ham/extracted/val.txt'
	voc_file = './data/henan_ham/voc.txt'

	data_x,data_y = loadDataFromFile(doc_file,voc_file)

	print(data_x.shape)
	print(data_y.shape)


if __name__ == '__main__':
	testloadDataFromFile()

	testEmbedding()
