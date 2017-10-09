#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-27 10:37:49
'''
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

import sys
reload(sys)
sys.path.append('../BaseUtil/')

import copy
from DeepLearningBaseModel import BaseModel


class RCNN(BaseModel):
	def __init__(self,num_classes,learning_rate,
		decay_rate,decay_steps,sequence_length,
		vocab_size,embed_size,hidden_size,context_size,
		l2_lambda,activation_func,grad_clip,
		initializer=tf.random_normal_initializer(stddev=0.1)):
		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.decay_rate = decay_rate
		self.decay_steps = decay_steps
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.context_size = context_size
		self.initializer = initializer
		self.activation = activation_func
		self.grad_clip = grad_clip
		self.l2_lambda = l2_lambda

		self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name = 'input_x')
		self.input_y = tf.placeholder(tf.int32,[None,self.num_classes],name = 'input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_keep_prob')
		self.batch_size = tf.shape(self.input_x)[0]

		self.global_step = tf.Variable(0,trainable = False,name = 'Global_step')
		
		self.instantiate_weights()

		self.logits = self.inference()

		self.loss_val = self.loss()

		self.train_op = self.train()
		
		self.predictions = tf.argmax(self.logits,axis = 1,name = 'predictions')

		correct_pred = tf.equal(self.predictions,tf.argmax(self.input_y,axis = 1))

		self.accuracy = tf.reduce_mean(tf.cast(correct_pred,'float'),name = 'accuracy')


	def instantiate_weights(self):
		with tf.name_scope('weight'):
			with tf.device('/cpu:0'):
				self.Embedding = tf.get_variable('Embedding',
					shape = [self.vocab_size,self.embed_size],initializer = self.initializer)
			
			self.W_l = tf.get_variable('W_l',shape = [self.context_size,self.context_size],
				initializer = self.initializer)

			self.W_r = tf.get_variable('W_r',shape = [self.context_size,self.context_size],
				initializer = self.initializer)

			self.W_sl = tf.get_variable('W_sl',shape = [self.embed_size,self.context_size],
				initializer = self.initializer)

			self.W_sr = tf.get_variable('W_sr',shape = [self.embed_size,self.context_size],
				initializer = self.initializer)

			self.W_projection = tf.get_variable('W_projection',
				shape = [self.context_size*2 + self.embed_size,self.num_classes],
				initializer = self.initializer)

			self.b_projection = tf.get_variable('bias',shape = [self.num_classes])

	def getContextLeft(self,context_left,embedding_left):
		'''
		context_previous:[bacth,context_size]
		embedding_previous:[bacth,embed_size]

		return: [bacth,context_size]
		'''

		left_c = tf.matmul(context_left,self.W_l) #[bacth context_size]
		left_e = tf.matmul(embedding_left,self.W_sl) #[bacth context_size]

		left_h = left_c + left_e #[batch context_size]

		context_left = self.activation(left_h) #[bacth context_size]

		return context_left
	
	def getContextRight(self,context_right,embedding_right):
		right_c = tf.matmul(context_right,self.W_r)
		right_e = tf.matmul(embedding_right,self.W_sr)

		right_h = right_c + right_e

		context_right = self.activation(right_h)

		return context_right


	def convertLayerWithRNN(self):
		#1. split input data
		#[sequence_length,bacth,1,embed_size]
		embedded_words_split = tf.split(self.embedded_words,self.sequence_length,axis = 1)
		
		#[sequence_length,bacth,embed_size]
		embedded_words_squeenzed = [tf.squeeze(x,axis = 1) for x in embedded_words_split] 
		
		#[batch embed_size]
		embedding_previous = tf.zeros([self.batch_size,self.embed_size])

		#[batch_size context_size]
		#context_previous = [self.left_side_first_word_context] * batch_size
		context_previous = tf.zeros([self.batch_size,self.context_size])

		context_left_list = []
		
		#curr_embedding_word: [bacth,embed_size]
		for idx,curr_embedding_word in enumerate(embedded_words_squeenzed):
			
			#[bacth context_size]
			context_left = self.getContextLeft(context_previous,embedding_previous) 
			context_left_list.append(context_left)
			
			embedding_previous = curr_embedding_word
			context_previous = context_left

		embedded_words_squeezed_reverse = copy.copy(embedded_words_squeenzed)
		embedded_words_squeezed_reverse.reverse()

		embedding_afterward = tf.zeros([self.batch_size,self.embed_size])
		context_afterward = tf.zeros([self.batch_size,self.context_size])

		context_right_list = []
		for idx ,curr_embedding_word in enumerate(embedded_words_squeezed_reverse):
			context_right = self.getContextRight(context_afterward,embedding_afterward)
			context_right_list.append(context_right)

			embedding_afterward = curr_embedding_word
			context_afterward = context_right


		
		context_right_list_cnt = len(context_right_list)

		context_output = [] #[sequence_length batch 2*context_size+embed_size]
		for idx,curr_embedding_word in enumerate(embedded_words_squeenzed):
			#[batch 2*context_size+embed_size]
			cand_representation = tf.concat([context_left_list[idx],
				curr_embedding_word,
				context_right_list[context_right_list_cnt-1-idx]],
				axis = 1) 

			context_output.append(cand_representation)

		#[batch sequence_length 2*context_size+embed_size]
		outputs = tf.stack(context_output,axis = 1)

		return outputs
	
	
	def inference(self):
		#[bacth,sequence_length,embed_size]
		self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x) 

		#[batch_size sequence_length 2*context_size+embed_size]
		context_presentation = self.convertLayerWithRNN() 

		#[batch_size 2*context_size+embed_size]
		output_pooling = tf.reduce_max(context_presentation,axis = 1) 

		with tf.name_scope('dropout'):
			h_drop = tf.nn.dropout(output_pooling,keep_prob = self.dropout_keep_prob)

		with tf.name_scope('projection'):
			#[bacth num_classes]
			logits = tf.nn.xw_plus_b(h_drop,self.W_projection,self.b_projection) 

		return logits
	
	def loss(self):
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y,
				logits = self.logits)

			loss = tf.reduce_mean(losses)

			l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])

			loss += l2_loss * self.l2_lambda

		return loss

	def train(self):
		learning_rate = tf.train.exponential_decay(self.learning_rate,
		self.global_step,self.decay_steps,
		self.decay_rate,staircase = True)

		#use grad_clip to hand exploding or vanishing gradients
		optimizer = tf.train.AdamOptimizer(learning_rate)
		grads_and_vars = optimizer.compute_gradients(self.loss_val)

		for idx ,(grad,var) in enumerate(grads_and_vars):
			if grad is not None:
				grads_and_vars[idx] = (tf.clip_by_norm(grad,self.grad_clip),var)

		train_op = optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)
			
		return train_op



def test():
	#below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
	num_classes=3
	learning_rate=0.01
	decay_steps=1000
	decay_rate=0.9
	sequence_length=5
	vocab_size=10000
	embed_size=100
	hidden_size = 100
	context_size = 128
	l2_lambda = 0.001
	activation_func = tf.tanh
	dropout_keep_prob=1#0.5
	grad_clip = 5.0 

	textRCNN = RCNN(num_classes, learning_rate,
		decay_rate,decay_steps,sequence_length,vocab_size,embed_size,
		hidden_size,context_size,l2_lambda,activation_func,grad_clip)

	batch_size=3
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
			input_y=input_y=np.array([[1,0,0],[0,0,1],[0,1,0]]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
			loss,acc,predict,_ = sess.run([textRCNN.loss_val,textRCNN.accuracy,
				textRCNN.predictions,textRCNN.train_op],
				feed_dict={textRCNN.input_x:input_x,
					textRCNN.input_y:input_y,
					textRCNN.batch_size:batch_size,
					textRCNN.dropout_keep_prob:dropout_keep_prob})

			print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)

#test()
