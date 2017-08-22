#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-08-11 08:15:33
'''
import abc

class BaseModel(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def instantiate_weights(self):
		return 


	@abc.abstractmethod
	def inference(self):
		return

	@abc.abstractmethod
	def loss(self):
		return

	@abc.abstractmethod
	def train(self):
		return

	
