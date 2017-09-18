#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-05 09:11:30
'''

import sys 
reload(sys)
sys.setdefaultencoding('utf8')

sys.path.append('/home/dengzhilong/code_for_learning/text_classification/BaseUtil/')
from DeepLearningBaseModel import BaseModel

import tensorflow as tf
import numpy as np

class FastTextModel(BaseModel):
	def __init__(self,vocab_size,embedding_size,num_classes,sequence_length,\
		learning_rate,decay_steps, decay_rate,l2_reg_lambda,is_training = True):
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.num_classes = num_classes
		self.sequence_length = sequence_length
		self.learning_rate = learning_rate
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.is_training = is_training
		self.l2_reg_lambda = l2_reg_lambda
		
		self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name = 'input_x')
		self.input_y = tf.placeholder(tf.int32,[None,self.num_classes],name = 'input_y')

		self.global_step = tf.Variable(0,trainable = False,name = 'global_step')
		
		self.instantiate_weights()

		self.logits = self.inference()
		self.loss_val = self.loss()
		
		self.train_op = self.train()

		self.predictions = tf.argmax(self.logits,axis = 1,name = 'predictions')
		
		correct_pred = tf.equal(tf.argmax(self.input_y,axis = 1),self.predictions)

		self.accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'),name = 'accuracy')

	def instantiate_weights(self):
		with tf.name_scope('word_embedding'):
			self.word_embedding_mat = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0),name = 'word_embedding')

			self.W = tf.Variable(tf.random_uniform([self.embedding_size,self.num_classes],-1.0,1.0),name = 'W')
			self.b = tf.Variable(tf.random_uniform([self.num_classes],-1.0,1.0),name = 'bias')
	

	def inference(self):
		'''
		1. word embedding
		2. average embedding
		3. linear classifier
		'''
		#word embedding
		#[batch,sequence_length,embedding_size]
		sentence_embedding = tf.nn.embedding_lookup(self.word_embedding_mat,self.input_x)

		#average embedding
		#[batch, embedding_size]
		self.sentence_embedding = tf.reduce_mean(sentence_embedding,axis = 1)

		logits = tf.matmul(self.sentence_embedding,self.W) + self.b

		return logits
	

	def loss(self):
		with tf.name_scope('loss'):
			data_loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y,
				logits = self.logits)

			data_loss = tf.reduce_mean(data_loss)

			l2_loss = tf.add_n([tf.nn.l2_loss(cand_var) for cand_var in tf.trainable_variables() if 'bias' not in  cand_var.name])

			data_loss += l2_loss * self.l2_reg_lambda

			return data_loss

	def train(self):
		with tf.name_scope('train'):
			learning_rate = tf.train.exponential_decay(self.learning_rate,
				self.global_step,self.decay_steps,self.decay_rate,staircase = True)

			train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step = self.global_step,\
				learning_rate = learning_rate,optimizer ='Adam')

		return train_op
		
