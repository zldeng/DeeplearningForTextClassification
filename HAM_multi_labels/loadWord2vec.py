#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-02-08 10:33:41
'''
import sys
import gensim

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
	embedding_size = word2vec_model.size()

	embedding_matrix = np.zeros((vocab_size,embedding_size))

	find_cnt = 0
	for w,idx word2idx:
		if idx >= vocab_size:
			continue

		if word in word2vec_model:
			find_cnt += 1
			embedding_matrix[idx] = word2vec_model[word]

	print('word in embedding matrix:',find_cnt)

	return embedding_matrix

if __name__ == '__main__':
	word2vec_model_file = ''
	voc_file = ''

	embedding_matrix = generateEmbeddingMatrixFromWord2vec(word2vec_model_file,voc_file)
