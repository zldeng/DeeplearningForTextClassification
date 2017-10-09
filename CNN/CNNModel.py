#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-22 09:16:08
'''
import tensorflow as tf
import numpy as np


import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('../BaseUtil/')
from DeepLearningBaseModel import BaseModel

class TextCNN(BaseModel):
	def __init__(self,sequence_length,num_classes,vocab_size,
		embeding_size,filter_sizes,num_filters,l2_reg_lambda,
		learning_rate,decay_steps,decay_rate):

		self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name = 'input_x')
		self.input_y = tf.placeholder(tf.int32,[None,num_classes],name = 'input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_keep_prob')

		self.batch_size = self.input_x.shape[0].value
		
		self.sequence_length = sequence_length
		self.num_classes = num_classes
		self.vocab_size = vocab_size
		self.embeding_size = embeding_size
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.l2_reg_lambda = l2_reg_lambda
		self.learning_rate = learning_rate
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate

		self.global_step = tf.Variable(0,trainable = False,name = 'global_step')
		#self.epoch_step = tf.Variable(0,trainable =False, name = 'epoch_step')
		#self.epoch_increment = tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))


		self.instantiate_weights()
		
		self.logits = self.inference()
		
		self.loss_val = self.loss()

		self.train_op = self.train()

		self.predictions = tf.argmax(self.logits,axis = 1,name = 'predictions')
		
		correct_prediction = tf.equal(self.predictions,tf.argmax(self.input_y,1))

		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'),name = 'accuracy')

	

	def instantiate_weights(self):
		with tf.device('/cpu:0'),tf.name_scope('embeding'):
			self.embedding = tf.Variable(tf.random_uniform([self.vocab_size,self.embeding_size],
				-1.0,1.0),name = 'embedding')
			self.embeded_chars = tf.nn.embedding_lookup(self.embedding,self.input_x)
			self.embeded_chars_expended = tf.expand_dims(self.embeded_chars,-1)

	def inference(self):
		'''
		1. converlution layer
		2. max_pooling
		3. dropout
		4. FC
		5. softmax
		'''
		pooled_outputs = []
		
		#filter_sizes is a int list. eg [1,2,3]
		for i,filter_size in enumerate(self.filter_sizes):
			with tf.name_scope('convo-maxpool-%s' % filter_size):
				filter_shape = [filter_size,self.embeding_size,1,self.num_filters]

				W = tf.Variable(tf.truncated_normal(filter_shape,stddev = 0.1), name = 'W')
				b = tf.Variable(tf.constant(0.1,shape = [self.num_filters]),name = 'b')

				#conv
				conv = tf.nn.conv2d(self.embeded_chars_expended,W,strides = [1,1,1,1],
					padding = 'VALID',name = 'conv')

				#non-linear
				h = tf.nn.relu(tf.nn.bias_add(conv,b),name = 'relu')

				#max_pooling
				pooled = tf.nn.max_pool(
					h,
					ksize = [1,self.sequence_length-filter_size+1,1,1],
					strides = [1,1,1,1],
					padding = 'VALID',
					name = 'maxpool')

				pooled_outputs.append(pooled)

		#combine pooled feature
		num_filters_total = len(self.filter_sizes) * self.num_filters

		self.h_pool = tf.concat(pooled_outputs,3)

		self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

		#dropout
		with tf.name_scope('dropout'):
			self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

		#FC
		with tf.name_scope('FC'):
			W = tf.get_variable('W',shape = [num_filters_total,self.num_classes],
				initializer = tf.contrib.layers.xavier_initializer())
			b = tf.Variable(tf.constant(0.1,shape = [self.num_classes]),name = 'b')

			self.score = tf.nn.xw_plus_b(self.h_drop,W,b,name = 'score')

			return self.score


	def loss(self):
		with tf.name_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y,
				logits = self.logits)

			data_loss = tf.reduce_mean(losses)

			l2_loss = tf.add_n([tf.nn.l2_loss(cand_var) for cand_var in tf.trainable_variables() if 'b' not in cand_var.name])

			data_loss = loss + l2_loss * self.l2_reg_lambda

			return data_loss
	

	def train(self):
		learning_rate = tf.train.exponential_decay(self.learning_rate,
			self.global_step,self.decay_steps,self.decay_rate,staircase = True)

		train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step = self.global_step,
			learning_rate = learning_rate,optimizer ='Adam')

		return train_op






#test started
def test():
	#below is a function test; if you use this for text classifiction, you need to tranform sentence to indices of vocabulary first. then feed data to the graph.
	num_classes=3
	learning_rate=0.01
	batch_size=3
	decay_steps=1000
	decay_rate=0.9
	sequence_length=5
	vocab_size=10000
	embed_size=100
	dropout_keep_prob=1#0.5
	l2_lambda = 0.1

	textRNN = TextCNN(sequence_length,num_classes,vocab_size,
		embed_size,filter_sizes=[3,4,5],num_filters = 128,l2_reg_lambda = 0.01,
		learning_rate = learning_rate,decay_steps = decay_steps,decay_rate = decay_rate)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
			input_y=input_y=np.array([[1,0,0],[0,0,1],[0,1,0]]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
			loss,acc,predict,_ = sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
			print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)


if __name__ == '__main__':
	test()
