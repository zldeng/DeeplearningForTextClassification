#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2018-02-07 17:41:39
'''
import sys
import json
import numpy as np

sentence_num_per_doc = 50
word_num_persentence = 30

def loadVocab(vocab_file):
	word2id = {}
	id2word = {}

	with open(vocab_file,'rb') as f:
		for line in f:
			line = line.decode()

			line_list = line.strip().split('\t')
			if len(line_list) != 2:
				print('Err_line:',line)
				continue

			word = line_list[0]
			word_idx = int(line_list[1])

			word2id[word] = word_idx
			id2word[word_idx] = word

	return word2id,id2word

def loadData():
	doc_file = ''
	vocab_file = ''
	
	word2id,id2word = loadVocab(vocab_file)

	doc_data = []
	doc_label = []

	with open(doc_file) as f:
		for line in f:
			line_list = line.strip().split('\t')

			if len(line_list) != 2:
				print 'Error line:',line
				continue

			sentence_list = line_list[0].strip().split(' ')
			label_list = line_list[1].split(',')
			


