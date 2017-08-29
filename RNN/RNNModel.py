#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-11 08:08:35
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

import sys
reload(sys)
sys.path.append('/home/dengzhilong/code_for_learning/text_classification/BaseUtil/')
from DeepLearningBaseModel import BaseModel


class TextRNN(BaseModel):
	def __init__(self,num_classes,learning_rate,decay_steps,decay_rate,\
			sequence_length,vocab_size,embed_size,hidden_size,is_training,\
			l2_lambda,grad_clip,
			initializer = tf.random_normal_initializer(stddev=0.1)):

		self.num_classes = num_classes
		self.learning_rate = learning_rate
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.sequence_length = sequence_length
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size
		self.is_training = is_training
		self.l2_lambda = l2_lambda
		self.grad_clip = grad_clip
		self.initializer = initializer
			

		#add palceholder
		self.input_x = tf.placeholder(tf.int32,[None,self.sequence_length],name = 'input_x')
		self.input_y = tf.placeholder(tf.int32,[None,num_classes],name = 'input_y')
		self.dropout_keep_prob = tf.placeholder(tf.float32,name = 'dropout_keep_prob')

		self.global_step = tf.Variable(0,trainable = False, name = 'Global_step')
		self.epoch_step = tf.Variable(0,trainable = False,name = 'Epoch_step')
		self.epoch_increment = tf.assign(self.epoch_step,tf.add(self.epoch_step,tf.constant(1)))
		

		self.instantiate_weights()
		self.logits = self.inference()
		
		self.loss_val = self.loss()
		self.train_op = self.train()
		self.predictions = tf.argmax(self.logits,axis = 1,name = 'predictions')
		
		#print self.predictions
		#print self.input_y

		correct_prediction = tf.equal(self.predictions,tf.argmax(self.input_y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'),name = 'accuracy')


	def instantiate_weights(self):
		'''define all weight'''
		with tf.name_scope('embedding'):
			self.Embedding = tf.get_variable('Embedding',shape = [self.vocab_size,self.embed_size],
				initializer = self.initializer)
			
			self.W_projection = tf.get_variable('W_projection',shape = [self.hidden_size * 2,self.num_classes],
			initializer = self.initializer)

			self.b_projection = tf.get_variable('b_projection',shape = [self.num_classes])

	def inference(self):
		'''
		1. embedding layer
		2. Bi-LSTM layer
		3. concat Bi-LSTM output
		4. FC(full connected) layer
		5. softmax layer
		'''

		#embedding layer
		with tf.device('/cpu:0'),tf.name_scope('embedding'):
			self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)

		#Bi-LSTM layer
		'''
		lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
		lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
		
		if self.dropout_keep_prob is not None:
			lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob = self.dropout_keep_prob)
			lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob = self.dropout_keep_prob)

		outputs,output_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,self.embedded_words,dtype = tf.float32)
		'''
		
		
		#BI-GRU layer
		gru_fw_cell = rnn.GRUCell(self.hidden_size)
		gru_bw_cell = rnn.GRUCell(self.hidden_size)

		if self.dropout_keep_prob is not None:
			gru_fw_cell = rnn.DropoutWrapper(gru_fw_cell,output_keep_prob = self.dropout_keep_prob)
			gru_bw_cell = rnn.DropoutWrapper(gru_bw_cell,output_keep_prob = self.dropout_keep_prob)

		outputs,output_states = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell,gru_bw_cell,self.embedded_words,dtype = tf.float32)
		
		#concat output
		#each output in outputs is [batch sequence_length hidden_size]
		
		#concat forward output and backward output
		output_cnn = tf.concat(outputs,axis = 2) #[batch sequence_length 2*hidden_size]
		
		output_cnn_last = tf.reduce_mean(output_cnn,axis = 1) #[batch_size,2*hidden_size]
				
		#FC layer
		with tf.name_scope('output'):
			self.score = tf.matmul(output_cnn_last,self.W_projection) + self.b_projection


		return self.score

	def loss(self):
		with tf.name_scope('loss'):
			#print self.input_y
			#print self.logits
			losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.input_y,logits = self.logits)

			loss = tf.reduce_mean(losses)

			l2_loss = tf.add_n([tf.nn.l2_loss(cand_v) for cand_v in tf.trainable_variables() if 'bias' not in cand_v.name]) * self.l2_lambda

			loss += l2_loss

		return loss

	def train(self):
		learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,
			self.decay_steps,self.decay_rate,staircase = True)

		#train_op = tf.contrib.layers.optimize_loss(self.loss_val,global_step = self.global_step,
		#	learning_rate = learning_rate,optimizer = 'Adam')
		
		#use grad_clip to hand exploding or vanishing gradients
		optimizer = tf.train.AdamOptimizer(learning_rate)
		grads_and_vars = optimizer.compute_gradients(self.loss_val)

		for idx ,(grad,var) in enumerate(grads_and_vars):
			if grad is not None:
				grads_and_vars[idx] = (tf.clip_by_norm(grad,self.grad_clip),var)

		train_op = optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)
			

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
	hidden_size = 100
	is_training=True
	dropout_keep_prob=1#0.5
	l2_lambda = 0.1

	textRNN=TextRNN(num_classes, learning_rate,
		decay_steps, decay_rate,sequence_length,vocab_size,embed_size,
		hidden_size,l2_lambda,is_training)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(100):
			input_x=np.zeros((batch_size,sequence_length)) #[None, self.sequence_length]
			input_y=input_y=np.array([[1,0,0],[0,0,1],[0,1,0]]) #np.zeros((batch_size),dtype=np.int32) #[None, self.sequence_length]
			loss,acc,predict,_ = sess.run([textRNN.loss_val,textRNN.accuracy,textRNN.predictions,textRNN.train_op],feed_dict={textRNN.input_x:input_x,textRNN.input_y:input_y,textRNN.dropout_keep_prob:dropout_keep_prob})
			print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)


if __name__ == '__main__':
	test()












