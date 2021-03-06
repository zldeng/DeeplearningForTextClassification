#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-11 10:50:43
'''

import sys

reload(sys)
sys.path.append('../BaseUtil')

import tensorflow as tf
import numpy as np
import os,time,datetime

#from RCNNModel import RCNN 
from RCNNModelWithLSTM import RCNN
from DataUtil import loadSklearnDataAndSplitTrainTest
from DataUtil import batch_iter


#configuration
tf.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.flags.DEFINE_integer("num_epochs",60,"embedding size")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size for training/evaluating.") #批处理的大小 32-->128

tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少

tf.flags.DEFINE_string("ckpt_dir","text_rcnn_checkpoint/","checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints',10,'save checkpoints count')

tf.flags.DEFINE_integer("sequence_length",300,"max sentence length")
tf.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.flags.DEFINE_integer('context_size',128,'word contenxt size')
tf.flags.DEFINE_integer('hidden_size',128,'cell output size')

tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 5, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_float('validation_percentage',0.1,'validat data percentage in train data')
tf.flags.DEFINE_integer('dev_sample_max_cnt',1000,'max cnt of validation samples, dev samples cnt too large will case high loader')

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_float('grad_clip',5.0,'grad_clip')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


tf.flags.DEFINE_string("train_data","/home/dengzhilong/code_from_my_git/data/all_hn_train_0817.sk",
	"path of traning data.")

tf.flags.DEFINE_string('tag2id_file','/home/dengzhilong/code_from_my_git/data/tag_level_1.data','label tag2id file')
tf.flags.DEFINE_string('tag_level','1','label tag level')

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()


for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')

sys.stderr.write('begin train....\n')
sys.stderr.write('begin load train data and create vocabulary...\n')

vocab_processor,train_data,dev_data = loadSklearnDataAndSplitTrainTest(FLAGS.tag2id_file,
	FLAGS.train_data,FLAGS.validation_percentage,FLAGS.dev_sample_max_cnt)

sys.stderr.write('load train data done\n')

x_train,y_train = train_data[0],train_data[1]
x_dev,y_dev = dev_data[0],dev_data[1]

with tf.Graph().as_default():
	sess_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	sess = tf.Session(config = sess_conf)

	with sess.as_default():
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_rcnn", timestamp))

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir,FLAGS.ckpt_dir))
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		checkpoints_prefix = os.path.join(checkpoint_dir,'model')

		vocab_processor.save(os.path.join(out_dir,'vocab'))
		
		
		rcnn = RCNN(num_classes = y_train.shape[1],
			learning_rate = FLAGS.learning_rate,
			decay_rate = FLAGS.decay_rate,
			decay_steps = FLAGS.decay_steps,
			sequence_length = x_train.shape[1],
			vocab_size = len(vocab_processor.vocabulary_),
			embed_size = FLAGS.embed_size,
			hidden_size = FLAGS.hidden_size,
			context_size = FLAGS.context_size,
			l2_lambda = FLAGS.l2_reg_lambda,
			activation_func = tf.tanh,
			grad_clip = FLAGS.grad_clip)


		saver = tf.train.Saver(tf.global_variables(),max_to_keep = FLAGS.num_checkpoints)
		
		sess.run(tf.global_variables_initializer())


		def train_step(x_batch,y_batch):
			feed_dict = {
				rcnn.input_x : x_batch,
				rcnn.input_y : y_batch,
				rcnn.dropout_keep_prob:FLAGS.dropout_keep_prob
				}

			tmp,step,loss,accuracy = sess.run([rcnn.train_op,
				rcnn.global_step,rcnn.loss_val,rcnn.accuracy],
				feed_dict)

			time_str = datetime.datetime.now().isoformat()
			print "{}:step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)
			

		def dev_step(x_batch,y_batch):
			feed_dict = {
				rcnn.input_x : x_batch,
				rcnn.input_y : y_batch,
				rcnn.dropout_keep_prob:1.0
				}
			

			step,loss,accuracy = sess.run([rcnn.global_step,rcnn.loss_val,
				rcnn.accuracy],feed_dict)
			
			time_str = datetime.datetime.now().isoformat()
			print "dev_result: {}:step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)

		
		for epoch_idx in range(FLAGS.num_epochs):
			batches = batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size) 
				
			for batch in batches:
				x_batch,y_batch = zip(*batch)
				
				train_step(x_batch,y_batch)

				current_step = tf.train.global_step(sess,rcnn.global_step)

			if epoch_idx % FLAGS.validate_every == 0:
				print '\n'
				dev_step(x_dev,y_dev)

			path = saver.save(sess,checkpoints_prefix,global_step=epoch_idx)
			print("Saved model checkpoint to {}\n".format(path))




































		

