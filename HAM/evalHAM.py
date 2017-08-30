#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 09:25:49
'''
import sys
import tensorflow as tf
import numpy as np

reload(sys)
import os
sys.path.append('../BaseUtil/')

from tensorflow.contrib import learn

from HAMDataUtil import loadTestDataAndConvertItToTensor
from DataUtil import batch_iter

tf.flags.DEFINE_string('checkpoint_dir','./runs_hn_bi_gru/1503976075/text_rnn_checkpoint/','the selected model for evaluated')
tf.flags.DEFINE_string('checkpoint_file','./runs_hn_bi_gru/1503976075/text_rnn_checkpoint/model-44','the selected model for evaluated')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer('batch_size','64','batch size')

tf.flags.DEFINE_integer('max_sentence_num',30,' max sentence cnt in doc')
tf.flags.DEFINE_integer('max_sentence_length',30,'max word cnt in a sentence')

tf.flags.DEFINE_string('tag2id_file','/home/dengzhilong/tensorflow/data/tag_level_1.data','label tag2id file')
tf.flags.DEFINE_string('tag_level','1','label tag level')

tf.flags.DEFINE_string('ham_test_file','/home/dengzhilong/tensorflow/data/henan_1th_all_labeled_data_from_excel.available.test.sklearn','ham format test file')

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()


for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')


#test model
#load test file and transfor
tag2id_file = FLAGS.tag2id_file

print 'load test data from sk_file'
vocab_file = os.path.join(FLAGS.checkpoint_dir,'../','vocab')

x_test,y_test = loadTestDataAndConvertItToTensor(FLAGS.test_ham_file,\
	FLAGS.tag2id_file,vocab_file,FLAGS.max_sentence_num,\
	FLAGS.max_sentence_length)

y_test = np.argmax(y_test,axis = 1)

print 'load test data done'

graph = tf.Graph()

all_predictions = []

with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	sess = tf.Session(config = session_conf)

	with sess.as_default():
		checkpoint_file = FLAGS.checkpoint_file
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

		saver.restore(sess,checkpoint_file)

		input_x = graph.get_operation_by_name('input_x').outputs[0]
		dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

		predictions = graph.get_operation_by_name('predictions').outputs[0]

		
		batches = batch_iter(list(x_test),FLAGS.batch_size,shuffle=False)

		for x_batch in batches:
			cand_predictions = sess.run(predictions,{input_x:x_batch,dropout_keep_prob:1.0})

			all_predictions = np.concatenate((all_predictions,cand_predictions))

print y_test[0]
print all_predictions[0]


correct_predictions = float(sum(all_predictions == y_test))

print 'all_test_rnn: ' + str(len(y_test))
print 'accuracy: ' + str(correct_predictions / len(y_test))




