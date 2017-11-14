#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-14 09:25:49
'''
import sys
import tensorflow as tf
import numpy as np
import pickle

reload(sys)
import os
sys.path.append('../BaseUtil/')

from tensorflow.contrib import learn
from sklearn.metrics import classification_report

from DataUtil import loadLabeledData
from DataUtil import batch_iter

#tf.flags.DEFINE_string('checkpoint_dir','./runs_hn_cnn/1503391363/text_cnn_checkpoint/','the checkpoint dir')
#tf.flags.DEFINE_string('checkpoint_file','runs_hn_cnn/1503391363/text_cnn_checkpoint/model-59','the selected model for evaluated')

tf.flags.DEFINE_string('checkpoint_dir','./runs_hn_cnn/1510106998/text_cnn_checkpoint/','the checkpoint dir')
tf.flags.DEFINE_string('checkpoint_file','./runs_hn_cnn/1510106998/text_cnn_checkpoint/model-59','the selected model for evaluated')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_integer('batch_size','64','batch size')

tf.flags.DEFINE_string('test_data','/home/dengzhilong/code_from_my_git/data/parser_engine/parser.model.test.tag2','test file')
tf.flags.DEFINE_string("label_encoder",'label_encoder','label encoder name')

tf.flags.DEFINE_string('test_result_file','./eval.res','test result')

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()


for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')


print 'load vocab data and create vocab_processor...'
vocab_path = os.path.join(FLAGS.checkpoint_dir,'../','vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
print 'load vocab processort done\n'

print 'load label encoder...'
label_encoder_name = os.path.join(FLAGS.checkpoint_dir,'../','label_encoder')
label_encoder = pickle.load(file(label_encoder_name,'rb'))
print 'load label encoder done\n'

print 'Load test file'
labeled_data_id,labeled_data_X,labeled_data_y = loadLabeledData(FLAGS.test_data)

y_test = np.array(label_encoder.transform(labeled_data_y))

x_test = np.array(list(vocab_processor.transform(labeled_data_X)))

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

print type(all_predictions)

correct_predictions = float(sum(all_predictions == y_test))

print 'all_test_rnn: ' + str(len(y_test))
print 'accuracy: ' + str(correct_predictions / len(y_test))


pred_label = label_encoder.inverse_transform(all_predictions.astype(int))

print classification_report(labeled_data_y,pred_label)

res_file = file(FLAGS.test_result_file,'w')
for cand_pair in zip(labeled_data_id,pred_label,labeled_data_y):
	res_file.write('\t'.join(cand_pair) + '\n')

res_file.close()
