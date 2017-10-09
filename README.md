# DeeplearningForTextClassification
  
  本项目将实现目前深度学习在文本分类上的一些经典模型。代码使用python实现，基于Google Tensorflow机器学习库以及sklearn机器学习库。
  
---
# 各模型对应的论文
## CNN  
基于CNN的文本分类方法对应的paper为 [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)  
基于CNN的文本分类实现可参考：[Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)  
CNN在NLP中的使用可参考[深度学习与自然语言处理之四：卷积神经网络模型（CNN）](http://blog.csdn.net/malefactor/article/details/50519566)  

## RNN
基于RNN的文本分类方法对应的paper为[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://www.ijcai.org/Proceedings/16/Papers/408.pdf)  

## HAM(Hierarchical Attention Network)
基于HAM模型的文本分类方法对应的paper为[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)  

## Fasttext
基于Fasttext的文本分类方法对应的paper为[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759) 
代码中仅仅实现了最简单的基于单词的词向量求平均，并未使用b-gram的词向量，所以文本分类效果低于facebook开源的的[facebook fasttext](https://github.com/facebookresearch/fastText)  
