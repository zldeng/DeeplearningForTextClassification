#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-29 17:30:12
'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import json
from tensorflow.contrib import learn


def loadDataFromTrainFile(train_file,max_sentence_num,max_sentence_length,\
	tag2id_file,dev_percentage,max_dev_sampe_cnt):
	'''
	load train data and then:
	 1. convert train data to matrix [doc_size,max_sentence_num,max_sentence_length]
	 2. create word to word_idx and return the dict
	
	train_file:
		eahc line is a doc
		tag + '\t' + [[word_11,word_12,word13,...],[word_21,word_22,word23,..]]
	
	'''
	tag2id = {}

	for line in file(tag2id_file):
		line = line.strip()

		if '' == line:
			continue
		print 'tag: ' + line
		tag2id[line] = len(tag2id)
	
	num_classes = len(tag2id)

	if num_classes < 2:
		print 'Error_tag_cnt: ' + str(num_classes)
		sys.exit(1)

	train_data = []
	
	line_num = 0
	doc_sentence_idx = []
	gold_label_list = []
	sentence_cnt = 0

	real_max_sentence_num = 0
	real_max_sentence_length = 0
	for line in file(train_file):
		line_list = line.strip().split('\t')
		line_num += 1

		if len(line_list) != 4:
			print 'Err ' + str(line_num)
			continue

		cand_tag = line_list[2]
		if cand_tag not in tag2id:
			print 'Err_tag: ' + cand_tag
			continue
		
		try:
			cand_tag_idx = tag2id[cand_tag]
			sent_list = json.loads(line_list[3])
		except Exception,e:
			print 'Data error:' + str(line_num)
			continue

		if not isinstance(sent_list,list):
			print 'Err_data: ' + str(line_num)
			continue
		
		if len(sent_list) > max_sentence_num:
			sent_list = sent_list[:max_sentence_num]
		
		if len(sent_list) > real_max_sentence_num:
			real_max_sentence_num = len(sent_list)

		try:
			doc_beg_idx = sentence_cnt

			for word_list in sent_list:
				if len(word_list.strip().split(' ')) > real_max_sentence_length:
					real_max_sentence_length = len(word_list.strip().split(' '))

				train_data.append(word_list)
				sentence_cnt += 1

			doc_end_idx = sentence_cnt
			
			#print doc_beg_idx,doc_end_idx
			doc_sentence_idx.append((doc_beg_idx,doc_end_idx))

			cand_label = [0] * num_classes
			cand_label[cand_tag_idx] = 1

			gold_label_list.append(cand_label)

		except Exception,e:
			print 'Err ' + str(line_num) + ' ' + str(e)
			sys.exit(1)
	
	final_max_sentence_length = min(real_max_sentence_length,max_sentence_length)
	final_max_sentence_num = min(real_max_sentence_num,max_sentence_num)
	
	print 'final_max: ', final_max_sentence_num,final_max_sentence_length
	vocab_processor = learn.preprocessing.VocabularyProcessor(final_max_sentence_length)

	train_word_idx_mat = np.array(list(vocab_processor.fit_transform(train_data)))

	all_data_x = []
	for (beg_idx,end_idx) in doc_sentence_idx:
		sent_cnt = end_idx - beg_idx 

		cand_data_mat = train_word_idx_mat[beg_idx:end_idx]
		
		#print np.shape(cand_data_mat),sent_cnt
		if sent_cnt < final_max_sentence_num:
			
			null_list = []
			for i in range(final_max_sentence_num - sent_cnt):
				null_list.append([0]*final_max_sentence_length)

			null_arr = np.array(null_list)
			#print 'null: ',np.shape(null_arr)
			cand_data_mat = np.concatenate((cand_data_mat,null_arr),axis = 0)
			

		all_data_x.append(cand_data_mat)
	

	x = np.array(all_data_x)
	y = np.array(gold_label_list)

	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(len(gold_label_list)))

	x_shuffled = x[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	dev_sample_index = max(-1 * int(len(y_shuffled) * dev_percentage),-1*max_dev_sampe_cnt)
	
	#x is [doc_size,max_sentence_num,max_sentence_length]
	#y is [doc_size,num_classes]
	x_train,x_dev = x[:dev_sample_index],x[dev_sample_index:]
	y_train,y_dev = y[:dev_sample_index],y[dev_sample_index:]


	return vocab_processor,(x_train,y_train),(x_dev,y_dev),num_classes,\
		final_max_sentence_length,final_max_sentence_num


