# -*- coding: utf-8 -*-

from bert_serving.client import BertClient
from sklearn.svm import SVC
from opencc import OpenCC
from scipy import spatial

cc = OpenCC('t2s')
bc = BertClient()

sents_str = ['���ѤѮ�u�n', ''���ѬO�a�Ѯ�]

sents_vec = bc.encode(sents_str)
print(sents_vec)

result = 1 - spatial.distance.cosine(sents_vec[0], sents_vec[1])
print(result)