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
sys.path.append('/home/dengzhilong/code_for_learning/text_classification/BaseUtil/')

from tensorflow.contrib import learn
from RNNModel import TextRNN

from DataUtil import loadSklearnDataForTensorFlow
from DataUtil import batch_iter

tf.flags.DEFINE_string('checkpoint_dir','/home/dengzhilong/code_for_learning/text_classification/a03_TextRNN/runs/1502439627/text_rnn_checkpoint/','model checkpoint dir')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer('batch_size','64','batch size')


tf.flags.DEFINE_string('tag2id_file','/home/dengzhilong/tensorflow/cnn_text_classicication_tf/cnn-text-classification-tf/tag_level_1.data','label tag2id file')
tf.flags.DEFINE_string('tag_level','1','label tag level')
tf.flags.DEFINE_string('sklearn_test_file','/home/dengzhilong/work/call_reason/data/relabled_data/henan_1th_relable/relabeld_data.henan.test.sklearn','sklearn format test file')

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()


for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')


#test model
#load test file and transfor
tag2id_file = FLAGS.tag2id_file

print 'load test data from sk_file'
x_raw,y_test = loadSklearnDataForTensorFlow(FLAGS.sklearn_test_file,FLAGS.tag_level,tag2id_file)

y_test = np.argmax(y_test,axis = 1)

print 'load test data done'

print 'load vocab data and create vocab_processor'

vocab_path = os.path.join(FLAGS.checkpoint_dir,'../','vocab')

vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

print 'load vocab processort done'

print len(x_raw),len(x_raw[0])

x_test = np.array(list(vocab_processor.transform(x_raw)))

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

graph = tf.Graph()

all_predictions = []

with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	sess = tf.Session(config = session_conf)

	with sess.as_default():
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))

		saver.restore(sess,checkpoint_file)

		input_x = graph.get_operation_by_name('input_x').outputs[0]
		dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

		predictions = graph.get_operation_by_name('predictions').outputs[0]

		
		batches = batch_iter(list(x_test),FLAGS.batch_size, 1, shuffle=False)

		for x_batch in batches:
			cand_predictions = sess.run(predictions,{input_x:x_batch,dropout_keep_prob:1.0})

			all_predictions = np.concatenate((all_predictions,cand_predictions))

print y_test[0]
print all_predictions[0]


correct_predictions = float(sum(all_predictions == y_test))

print 'all_test_rnn: ' + str(len(y_test))
print 'accuracy: ' + str(correct_predictions / len(y_test))




