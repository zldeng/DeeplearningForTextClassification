#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-11 10:36:08
'''


import sys
import json
import numpy as np

from tensorflow.contrib import learn

reload(sys)
sys.setdefaultencoding('utf8')

def loadTag2Id(tag2id_file):
	tag2id = {}
	
	with file(tag2id_file) as tag_file:
		for line in tag_file:
			tag2id[line.strip()] = len(tag2id)

	print 'tag_cnt: ' + str(len(tag2id))
	return tag2id


def loadSklearnDataForTensorFlow(sklearn_file,tag2id_file):
	'''
		train and test must use the same tag2id_file
	'''

	tag2id = loadTag2Id(tag2id_file)

	x_text = []
	y_text = []

	tag_size = len(tag2id)

	with file(sklearn_file) as sk_in:
		for line in sk_in:
			line_list = line.strip().split(' ')

			if len(line_list) < 5:
				continue

			cand_tag = line_list[2]

			if cand_tag not in tag2id:
				continue

			cand_tag_idx = tag2id[cand_tag]

			cand_x = ' '.join(line_list[3:])
			cand_y = [0] * tag_size
			cand_y[cand_tag_idx] = 1

			x_text.append(cand_x)
			y_text.append(cand_y)

	return [x_text,np.array(y_text)]


def loadSklearnDataAndSplitTrainTest(tag2id_file,sklearn_data_file,dev_percentage,max_dev_sampe_cnt):
	'''
	load sklearn data and return tensorflow input data
	sk_file: file_name case_id tag_1 tag2 word_1 word_2 ...
	tag2id: 
		tag1 1
		tag2 2
		...
	return: vocab_processor,(x_train,y_train),(x_dev,y_dev)

	'''
	x_text,y = loadSklearnDataForTensorFlow(sklearn_data_file,tag2id_file)

	max_document_length = max([len(x.split(' ')) for x in x_text])

	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

	x = np.array(list(vocab_processor.fit_transform(x_text)))

	np.random.seed(10)

	shuffle_indices = np.random.permutation(np.arange(len(y)))

	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]
	
	dev_sample_index = max(-1 * int(len(y_shuffled) * dev_percentage),-1*max_dev_sampe_cnt)

	x_train,x_dev = x[:dev_sample_index],x[dev_sample_index:]
	y_train,y_dev = y[:dev_sample_index],y[dev_sample_index:]

	return vocab_processor,(x_train,y_train),(x_dev,y_dev)




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
