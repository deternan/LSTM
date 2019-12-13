# coding=utf8

'''
version: December 12, 2019 02:45 PM
Last revision: December 12, 2019 03:15 PM

Author : Chao-Hsuan Ke
'''

import re

# read data
filePath = 'D:\\Phelps\\GitHub\\Python\\LSTM\\data\\';        # file path
fileName = 'alllyric_3.txt';        # file name
f = open(filePath + fileName, 'r', encoding="utf-8")

# regular expression
r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
r2 = '[\\s]{2,20}'

# output data
outputFolder = filePath                 # output folder name
outputFile = 'alllyric_4 Chinese.txt'              # output file name
fp = open(outputFolder + outputFile, "a", encoding='UTF-8')       # 'a' --> overlapping


# read each lines
for line in f:
    newStr = re.sub(r1, '', line)
    resultStr = re.sub(r2, '', newStr)
    fp.writelines(resultStr.strip()+'\n')
    print(resultStr.strip())


f.close()
fp.close()