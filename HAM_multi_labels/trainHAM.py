#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-29 18:42:57
  version:v1.0
'''
 
import sys
import tensorflow as tf
import numpy as np
import os,time,datetime
import math
import pickle
from HAMModel import HAM
from loadData import batch_iter
from loadData import loadDataFromFile

#configuration
tf.flags.DEFINE_float("learning_rate",0.01,"learning rate")
tf.flags.DEFINE_integer("num_epochs",256,"embedding size")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluation.")
tf.flags.DEFINE_integer("decay_steps", 12000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.9, "Rate of decay for learning rate.") #0.5一次衰减多少

tf.flags.DEFINE_string("ckpt_dir","text_ham_checkpoint/","checkpoint location for the model")
tf.flags.DEFINE_integer('num_checkpoints',100,'save checkpoints count')

tf.flags.DEFINE_integer("embedding_size",128,"embedding size")
tf.flags.DEFINE_integer('hidden_size',128,'cell output size')

tf.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")

tf.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") 

tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

tf.flags.DEFINE_float("l2_reg_lambda", 0.0001, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float('grad_clip',4.0,'grad_clip')

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string("word2vec_model_file",
		"/home/dengzhilong/code/HAM_multi_labels/data/henan_ham/w2v_model_STD_NOSEG.vec",
		"pre-train word2vec")

tf.flags.DEFINE_string("vocabulary_file",
		"/home/dengzhilong/code/HAM_multi_labels/data/henan_ham/voc.txt",
		"word vocabulary file")

tf.flags.DEFINE_string("train_file",
		"/home/dengzhilong/code/HAM_multi_labels/data/henan_ham/extracted/train.txt",
		"train file")

tf.flags.DEFINE_string("val_file",
		"/home/dengzhilong/code/HAM_multi_labels/data/henan_ham/extracted/val.txt",
		"validation file")
tf.flags.DEFINE_float('pred_threshold',0.5,'pred threshols')

def evaluation(predictions,input_y):	
	#change tensor dtype to int32 and convert tensor to np.array
	#print('evaludate:',np.shape(predictions),np.shape(input_y))
	
	'''
	with open('tmp_pred','w') as f:
		import json
		f.write(json.dumps(predictions))
	'''

	def eval(predictions,input_y):
		precision_total, recall_total, correct = (0, 0, 0)

		sample_cnt = np.shape(input_y)[0]
		labels_cnt = np.shape(input_y)[1]

		for sample_idx in range(sample_cnt):
			for label_idx in range(labels_cnt):
				if predictions[sample_idx][label_idx] == 1:
					recall_total += 1

				if input_y[sample_idx][label_idx] == 1:
					precision_total += 1

				if input_y[sample_idx][label_idx] == 1\
					and input_y[sample_idx][label_idx] == predictions[sample_idx][label_idx]:
					correct += 1
		
		if precision_total == 0:
			precision_total = 1e-7
		
		if precision_total == 0:
			precison = -1
		else:
			precison = round((correct * 1.0 / precision_total),3)

		if recall_total == 0:
			recall = 0
		else:
			recall = round((correct * 1.0/recall_total),3)

		if precison * recall == 0:
			f_score = 0.0
		else:
			f_score = (2 * precison * recall)/(precison + recall)

		return (precison,recall,f_score)
	
	res = {}
	for pred_threshold in [0.5,0.6,0.7]:	
		predictions[predictions > pred_threshold] = 1
		predictions[predictions <= pred_threshold] = 0
		
		eval_res = eval(predictions,input_y)
		
		res[pred_threshold] = eval_res

	return res
		


#change here
tf.flags.DEFINE_integer('max_sentence_num',20,'max sentence num in a doc')
tf.flags.DEFINE_integer('max_sentence_length',50,'max word count in a sentence')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

vocab_size = 1351
tf.flags.DEFINE_integer('num_classes',6,"labels count")
data_dir = '/home/dengzhilong/code/HAM_multi_labels/cr_multi_data_tag_2/'
train_cr_multi_data = data_dir + 'cr_train_tag2_top4.data'
val_cr_multi_data = data_dir + 'cr_val_tag2_top4.data'

#vocab_size = 1508
#tf.flags.DEFINE_integer('num_classes',3,"labels count")
#data_dir = '/home/dengzhilong/code/HAM_multi_labels/cr_multi_data_tag_1/'
#train_cr_multi_data = data_dir + 'cr_train.data'
#val_cr_multi_data = data_dir + 'cr_val.data'

FLAGS=tf.flags.FLAGS
FLAGS._parse_flags()

for attr,value in sorted(FLAGS.__flags.items()):
	sys.stderr.write("{}={}".format(attr,value) + '\n')

print('Load train data ...')
#x_train,y_train = loadDataFromFile(FLAGS.train_file,FLAGS.vocabulary_file,
#		FLAGS.max_sentence_num,FLAGS.max_sentence_length)
#
#x_dev,y_dev = loadDataFromFile(FLAGS.val_file,FLAGS.vocabulary_file,
#		FLAGS.max_sentence_num,FLAGS.max_sentence_length)


with open(train_cr_multi_data,'rb') as f:
	x_train,y_train = pickle.load(f)

with open(val_cr_multi_data,'rb') as f:
	x_dev,y_dev = pickle.load(f)

print('Load data Done')
print('train data shape: ',np.shape(x_train),np.shape(y_train))
print ('dev data shape',np.shape(x_dev),np.shape(y_dev))

print(np.bincount(np.argmax(y_train,axis = 1)))
print(np.bincount(np.argmax(y_dev,axis = 1)))



sys.stdout.flush()


with tf.Graph().as_default():
	sess_conf = tf.ConfigProto(
		allow_soft_placement=FLAGS.allow_soft_placement,
		log_device_placement=FLAGS.log_device_placement)

	sess_conf.gpu_options.per_process_gpu_memory_fraction = 0.9

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

		ham = HAM(vocab_size,FLAGS.embedding_size,
				FLAGS.max_sentence_num,
				FLAGS.max_sentence_length,
				FLAGS.num_classes,
				FLAGS.hidden_size,
				FLAGS.learning_rate,
				FLAGS.decay_rate,
				FLAGS.decay_steps,
				FLAGS.l2_reg_lambda,
				FLAGS.grad_clip,
				is_training = True)

		saver = tf.train.Saver(tf.global_variables(),max_to_keep = FLAGS.num_checkpoints)

		sess.run(tf.global_variables_initializer())

		def train_step(x_batch,y_batch):
			feed_dict = {
				ham.input_x : x_batch,
				ham.input_y : y_batch,
				ham.dropout_keep_prob : FLAGS.dropout_keep_prob
			}

			tmp,step,loss,pred_sigmoid,func_loss,\
				l2_loss,pred_logits,pred_sigmoid,doc_vec = sess.run([ham.train_op,
						ham.global_step,
						ham.loss_val,
						ham.pred_sigmoid,
						ham.func_loss,
						ham.l2_loss,
						ham.logits,
						ham.pred_sigmoid,
						ham.doc_vec],
						feed_dict = feed_dict)
			
			'''
			print('\ntrain_y\n',y_batch)
			
			print('\npred_logits:\n',pred_logits)

			print('\npred_sigmoid:\n',pred_sigmoid)

			print('\ndoc_vec:\n',doc_vec)
			'''

			#eval_res = evaluation(pred_sigmoid,y_batch)
			time_str = datetime.datetime.now().isoformat()
			print("train_info\t{}:step {},loss {:g} ,func_loss {:g} ,l2_loss {:g}".format(time_str,step,loss,func_loss,l2_loss))
			
			sys.stdout.flush()


		def dev_step(dev_x,dev_y):
			sample_cnt = np.shape(dev_y)[0]
			batch_num = math.ceil(sample_cnt * 1.0 / FLAGS.batch_size)
			pred_y_sigmoid = []
			for batch_idx in range(batch_num):
				beg_idx = FLAGS.batch_size * batch_idx
				end_idx = min(FLAGS.batch_size * (1 + batch_idx),sample_cnt)

				if beg_idx < end_idx:
					sub_dev_x = dev_x[beg_idx:end_idx]
					sub_dev_y = dev_y[beg_idx:end_idx]


					feed_dict = {
						ham.input_x : sub_dev_x,
						ham.input_y : sub_dev_y,
						ham.dropout_keep_prob:1.0
						}

					step,loss,pred_sigmoid = sess.run([ham.global_step,
							ham.loss_val,
							ham.pred_sigmoid],
							feed_dict = feed_dict)
					
					if [] == pred_y_sigmoid:
						pred_y_sigmoid = pred_sigmoid
					else:
						pred_y_sigmoid = np.concatenate((pred_y_sigmoid,pred_sigmoid))

			time_str = datetime.datetime.now().isoformat()
			eval_res = evaluation(pred_y_sigmoid,dev_y)

			print("dev_result:{}:step {}, loss {:g}, eval_res {}".format(time_str,step,loss,eval_res))
			sys.stdout.flush()
			#sys.exit(1)

		for epoch_idx in range(FLAGS.num_epochs):
			print('begin epoch %d/%d:' % (epoch_idx,FLAGS.num_epochs))
			batches = batch_iter(list(zip(x_train,y_train)),FLAGS.batch_size) 
				
			for batch in batches:
				x_batch,y_batch = zip(*batch)
				
				train_step(x_batch,y_batch)

				current_step = tf.train.global_step(sess,ham.global_step)

			print("evaluation on dev data")
			dev_step(x_dev,y_dev)

			path = saver.save(sess,checkpoints_prefix,global_step=epoch_idx)
			print("Saved model checkpoint to {}\n".format(path))



















