# -*- coding: utf-8 -*-

'''
version: December 23 2019, 01:00 PM
Last revision: December 23 2019, 05:44 PM

Author : Chao-Hsuan Ke
'''

import os
import time
import numpy as np
from datetime import timedelta

from cnews_loader import read_vocab, read_category, batch_iter, process_output, build_vocab

base_dir = 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\topic classification\\cnn-rnn\\data\\'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

# output parameters
outputFolder = base_dir  # output folder name
trainoutput = "train_value_output.txt"  # output file name (training data)
valoutput = "validation_value_output.txt"  # output file name (validation data)
testoutput = "test_value_output.txt"  # output file name (testing data)


def merge(data, category, dataName):
    merageArray = np.append(data, category, axis=1)
    # save
    np.savetxt(base_dir + dataName, merageArray, fmt='%2.0f')


def getCategory(data, category, dataName):
    categoryType = np.zeros(shape=(len(category), 1))
    kk = 0
    for i in category:
        index = np.argwhere(i == np.max(i, axis=0))
        categoryType[kk] = index
        kk += 1
    #print(categoryType)
    merge(data, categoryType, dataName)

def outputTrain():
    print("output training data...")
    start_time = time.time()
    x_train, y_train = process_output(train_dir, word_to_id, cat_to_id, max_length=600)
    # get category and merge
    getCategory(x_train, y_train, trainoutput)
    # save
    # np.savetxt(base_dir + trainoutput, x_train)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def outputValidation():
    print("output validation data...")
    start_time = time.time()
    x_val, y_val = process_output(val_dir, word_to_id, cat_to_id, max_length=600)
    # get category and merge
    getCategory(x_val, y_val, valoutput)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def outputTest():
    print("output testing data...")
    start_time = time.time()
    x_test, y_test = process_output(test_dir, word_to_id, cat_to_id, max_length=600)
    # get category and merge
    getCategory( x_test, y_test, testoutput)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)

# output testing
outputValidation()
outputTrain()
outputTest()