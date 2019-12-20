#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Last revision: December 20 2019, 06:06 PM

Author : Chao-Hsuan Ke
'''

"""
将文本整合到 train、test、val 三个文件中
"""

import os

# dataset folder
originalPath = 'D:\\data model\\THUCTC\\THUCNews_tinydata\\'
# target files folder
cnewstrain = 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\topic classification\\cnn-rnn\\data\\cnews.train.txt'
cnewstest = 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\topic classification\\cnn-rnn\\data\\cnews.test.txt'
cnewsval = 'D:\\Phelps\\GitHub\\Python\\NeuralNetwork\\Text classification\\topic classification\\cnn-rnn\\data\\cnews.val.txt'

def _read_file(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')

def save_file(dirname):

    f_train = open(cnewstrain, 'w', encoding='utf-8')
    f_test = open(cnewstest, 'w', encoding='utf-8')
    f_val = open(cnewsval, 'w', encoding='utf-8')
    for category in os.listdir(dirname):    # 分类目录
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '\t' + content + '\n')
            elif count < 6000:
                f_test.write(category + '\t' + content + '\n')
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    #save_file('data/thucnews')
    save_file(originalPath)
    print(len(open(cnewstrain, 'r', encoding='utf-8').readlines()))
    print(len(open(cnewstest, 'r', encoding='utf-8').readlines()))
    print(len(open(cnewsval, 'r', encoding='utf-8').readlines()))

