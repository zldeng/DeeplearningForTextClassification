#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-29 18:42:57
'''
 
import sys
reload(sys)
sys.path.append('../BaseUtil')

import tensorflow as tf
import numpy as np
import os,time,datetime

from HAMModel import HAM
from DataUtil import batch_iter
from HAMDataUtil import loadDataFromTrainFile


#configuration
tf.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.flags.DEFINE_integer("num_epochs",30,"embedding size")
tf.flags.DEFINE_integer("batch_size", 100, "Batch size for training/evaluating.") #批处理的大小 32-->128

tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少

tf.flags.DEFINE_string("ckpt_dir","text_ham_checkpoint/","checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints',20,'save checkpoints count')

tf.flags.DEFINE_integer('max_sentence_num',30,'max sentence num in a doc')
tf.flags.DEFINE_integer('max_sentence_length',30,'max word count in a sentence')
tf.flags.DEFINE_integer("embedding_size",128,"embedding size")
tf.flags.DEFINE_integer('hidden_size',128,'cell output size')

tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.flags.DEFINE_float('validation_percentage',0.1,'validat data percentage in train data')

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")

tf.flags.DEFINE_float('grad_clip',5.0,'grad_clip')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


#tf.flags.DEFINE_string("train_data",\
#	"/home/dengzhilong/work/call_reason/data/all_data/all_hn_train_0817.sk",\
#	"path of traning data.")

tf.flags.DEFINE_string("train_file",\
	"/home/dengzhilong/tensorflow/data/ham_data/all_train.data.tag1.ham",\
	"path of traning data.")

tf.flags.DEFINE_string('tag2id_file',\
	'/home/dengzhilong/tensorflow/data/ham_data/tag_level_1.data',\
	'label tag2id file')

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()


for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')

sys.stderr.write('begin train....\n')
sys.stderr.write('begin load train data and create vocabulary...\n')

vocab_processor,train_data,dev_data,num_classes,\
	final_max_sentence_length,final_max_sentence_num \
		= loadDataFromTrainFile(FLAGS.train_file,\
			FLAGS.max_sentence_num,FLAGS.max_sentence_length,\
			FLAGS.tag2id_file,FLAGS.validation_percentage)

vocab_size = len(vocab_processor.vocabulary_)

print 'vocab_size: ' + str(vocab_size)
print 'Load train data done!'


x_train,y_train = train_data[0],train_data[1]
x_dev,y_dev = dev_data[0],dev_data[1]


print 'train_data:',np.shape(x_train),np.shape(y_train)
print 'dev_data:',np.shape(x_dev),np.shape(y_dev)

sys.stdout.flush()

with tf.Graph().as_default():
	sess_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	sess = tf.Session(config = sess_conf)

	with sess.as_default():
		timestamp = str(int(time.time()))
		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_ham", timestamp))

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir,FLAGS.ckpt_dir))
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		
		checkpoints_prefix = os.path.join(checkpoint_dir,'model')

		vocab_processor.save(os.path.join(out_dir,'vocab'))
	
		ham = HAM(vocab_size,\
			final_max_sentence_num,final_max_sentence_length,\
			num_classes,FLAGS.embedding_size,FLAGS.hidden_size,\
			FLAGS.learning_rate,FLAGS.decay_rate,FLAGS.decay_steps,\
			FLAGS.l2_reg_lambda,FLAGS.grad_clip,True)

		saver = tf.train.Saver(tf.global_variables(),max_to_keep = FLAGS.num_checkpoints)

		sess.run(tf.global_variables_initializer())

		def train_step(x_batch,y_batch):
			feed_dict = {
				ham.input_x : x_batch,
				ham.input_y : y_batch,
				ham.dropout_keep_prob : FLAGS.dropout_keep_prob
			}

			tmp,step,loss,accuracy = sess.run([ham.train_op,ham.global_step,ham.loss_val,ham.accuracy],feed_dict)

			time_str = datetime.datetime.now().isoformat()
			print "{}:step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)


		def dev_step(dev_x,dev_y):
			feed_dict = {
				ham.input_x : dev_x,
				ham.input_y : dev_y,
				ham.dropout_keep_prob:1.0
				}
			

			step,loss,accuracy = sess.run([ham.global_step,ham.loss_val,ham.accuracy],feed_dict)
			
			time_str = datetime.datetime.now().isoformat()
			print "dev_result: {}:step {}, loss {:g}, acc {:g}".format(time_str,step,loss,accuracy)


		for epoch_idx in range(FLAGS.num_epochs):
			batches = batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size) 
				
			for batch in batches:
				x_batch,y_batch = zip(*batch)
				
				train_step(x_batch,y_batch)

				current_step = tf.train.global_step(sess,ham.global_step)

			if epoch_idx % FLAGS.validate_every == 0:
				print '\n'
				dev_step(x_dev,y_dev)

			path = saver.save(sess,checkpoints_prefix,global_step=epoch_idx)
			print("Saved model checkpoint to {}\n".format(path))



















